
import os
import shutil
import concurrent.futures
import multiprocessing as mp
from config import SystemConfig
from vector_db import VectorDBFactory
from memory_builder import MemoryBuilder
from loguru import logger
from datetime import datetime

from dataset import load_locomo_dataset, load_longmemeval_dataset, train_valid_test_split, load_membench_dataset, load_hotpotqa_dataset, load_narrativeqa
from llm_api import OpenAIAPIClient, LocalClient, VLLMClient, LLMAPIClientBase
from memory_formation import MemoryFormation
from memory_update import MemoryUpdate
from memory_retrieval import MemoryRetriever
from trajectory_logger import QATrajectoryLog, get_collector
from utils import seed_everything


worker_builder = None
worker_retrieval = None
worker_collector = None
main_config = None

def init_worker(gpu_queue, config_from_main):
    global worker_builder, worker_retrieval, worker_collector, main_config
    import os
    from loguru import logger
    main_config = config_from_main
    
    
    
    logger.add(main_config.log_path, level="DEBUG", enqueue=True)
    try:
        gpu_id = gpu_queue.get(timeout=5)
    except:
        gpu_id = 0 
        
    logger.info(f"Worker process {os.getpid()} initializing on GPU {gpu_id}")
    model_path = main_config.llm.local_model_path if os.path.exists(main_config.llm.local_model_path) else main_config.llm.local_model
    device_str = f"cuda:{gpu_id}"
    try:
        api_client = VLLMClient(model_name_or_path=model_path, device=device_str)
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to load model: {e}")
        return

    
    try:
        vector_db = VectorDBFactory.create_db(main_config.vector_db)
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to connect to DB: {e}")
        return

    llm_executor = api_client
    formation_module = MemoryFormation(llm_executor=llm_executor)
    update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
    retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=main_config)
    
    worker_builder = MemoryBuilder(
        vector_db=vector_db, 
        formation=formation_module, 
        update=update_module,       
        config=main_config
    )
    worker_retrieval = retrieval_module
    worker_collector = get_collector(main_config.traj_dir)

def process_sample_logic(i, sample):
    global worker_builder, worker_retrieval, worker_collector, main_config
    
    current_builder = worker_builder
    current_retrieval = worker_retrieval
    current_collector = worker_collector
    
    current_config = main_config
    
    if current_builder is None:
        
        
        if 'builder' in globals():
            current_builder = globals()['builder']
            current_retrieval = globals()['retrieval_module']
            current_collector = globals()['collector']
            current_config = globals()['main_config']
    
    if current_builder is None or current_retrieval is None or current_collector is None or current_config is None:
        logger.error(f"Worker {os.getpid()}: Builder, Retrieval, Collector, or Config not initialized!")
        return
    
    sample_id = sample['qa'][0].get('sample_id') if sample['qa'] else f"sample_{i}"
    logger.info(f"Processing Sample: {sample_id} (PID: {os.getpid()})")

    if current_config.vector_db.from_scratch:
        current_builder.build_from_sample(sample)
    logger.success(f"Finish Processing Sample: {sample_id}")
    if sample_id == "03690ff2154ebecb454ea267cec529358ab6923b":
        print("skip this sample")
        return
    
    if not current_config.vector_db.from_scratch:
        answer_turn = 1
    else:
        answer_turn = 1

    for _ in range(answer_turn):
        for j, qa in enumerate(sample['qa']):
            question = qa['question']
            gold_answer = qa['answer']
            evidence = qa['evidence']
            logger.info(f"Retrieving for Q: {question}")
                    
            result = current_retrieval.retrieve_and_answer(question, sample_id=sample_id, category=qa.get('category', ''))
                    
            pred_answer = result['answer']
            traces = result['traces']

            current_collector.log_qa_step(QATrajectoryLog(sample_id=sample_id,
                                                        qa_id=f"{sample_id}_{j}",
                                                        question=question,
                                                        pred=pred_answer,
                                                        gold=gold_answer,
                                                        evidence=evidence,
                                                        traces=traces,
                                                        category=qa.get('category', '')))
            logger.success(f"Pred: {pred_answer} | Gold: {gold_answer}")

def process_sample_wrapper(args):
    return process_sample_logic(*args)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    
    config = SystemConfig()
    seed_everything(seed=config.seed)
    vdb_config = config.vector_db

    benchmark_name = config.data_name if hasattr(config, "data_name") else "longmemeval"
    DB_PATH = os.path.join(config.vector_db.path, benchmark_name+f"{datetime.now().strftime('%Y%m%d_%H%M')}")
    vdb_config.path = DB_PATH

    USE_LOCAL_LLM = config.USE_LOCAL_LLM
    USE_PARALLEL = config.USE_PARALLEL
    NUM_WORKERS = config.NUM_WORKERS

    logger.add(config.log_path, level="WARNING", enqueue=True)
    logger.info("Starting Memory Building Process")
    logger.info(f"Config: {config}")

    
    print("\nLoading dataset...")
    if config.data_name == "locomo":
        chat_data = load_locomo_dataset(config.dataset_path)
        train_data = [chat_data[0]]
        valid_data = [chat_data[1]]
        test_data = chat_data[2:]
    elif config.data_name == "hotpotqa":
        chat_data = load_hotpotqa_dataset(config.dataset_path)
        train_data, valid_data, test_data = train_valid_test_split(chat_data, seed=config.seed)
        test_data = test_data[0:72]
    else:
        raise ValueError(f"Unsupported dataset: {config.data_name}")
    if config.mode == "valid":
        data = valid_data
    elif config.mode == "test":
        data = test_data
    else:
        data = train_data
    
    if USE_LOCAL_LLM and USE_PARALLEL:
        
        print(f"Parallel processing mode (ProcessPoolExecutor, Workers: {NUM_WORKERS})")
        print("Using multiple processes to load local models to different GPUs. Please use nvidia-smi to monitor GPU memory.")
        manager = mp.Manager()
        gpu_queue = manager.Queue()
        
        for i in range(NUM_WORKERS):
            gpu_queue.put(i % 8) 
            
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=NUM_WORKERS, 
            initializer=init_worker, 
            initargs=(gpu_queue, config)
        )
        futures = {}
        try:
            print(f"Dispatching {len(data)} tasks...")
            for i, sample in enumerate(data):
                future = executor.submit(process_sample_wrapper, (i, sample))
                futures[future] = i

            print("Tasks started, waiting for results...")
            from concurrent.futures import as_completed
            
            for future in as_completed(futures):
                sample_idx = futures[future]
                try:
                    future.result() 
                    print(f"Sample {sample_idx} processed") 
                except Exception as exc:
                    logger.error(f"Sample {sample_idx} processing exception: {exc}")
        except KeyboardInterrupt:
            print("KeyboardInterrupt: User interrupted processing.")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            print("Forcing shutdown of Worker processes...")
            executor.shutdown(wait=False)

    elif not USE_LOCAL_LLM and USE_PARALLEL:
        
        print(f"Parallel processing mode (ThreadPoolExecutor, Workers: {NUM_WORKERS})")
        try:
            vector_db = VectorDBFactory.create_db(vdb_config)
            print("Vector database connection successful.")
        except Exception as e:
            print(f"Failed to connect to vector database: {e}")
            exit()

        print(f"Configuring remote LLM ({config.llm.strong_model})")
        api_key = os.environ.get("OPENAI_API_KEY") or config.llm.strong_api_key
        api_client = OpenAIAPIClient(model=config.llm.strong_model)
        
        llm_executor = api_client
        formation_module = MemoryFormation(llm_executor=llm_executor)
        update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
        retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=config)
        
        builder = MemoryBuilder(
            vector_db=vector_db, 
            formation=formation_module, 
            update=update_module,       
            config=config
        )
        collector = get_collector(config.traj_dir)
        globals()['builder'] = builder
        globals()['retrieval_module'] = retrieval_module
        globals()['collector'] = collector
        globals()['main_config'] = config
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            try:
                list(executor.map(process_sample_wrapper, enumerate(data)))
            except KeyboardInterrupt:
                print("User interrupted")
            

    else:
        print("Sequential processing mode")
        vector_db = VectorDBFactory.create_db(vdb_config)
        if USE_LOCAL_LLM:
            model_path = config.llm.local_model_path if os.path.exists(config.llm.local_model_path) else config.llm.local_model
            api_client = VLLMClient(model_name_or_path=model_path)
        else:
            api_client = OpenAIAPIClient(model=config.llm.strong_model)
        llm_executor = api_client
        formation_module = MemoryFormation(llm_executor=llm_executor)
        update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
        retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=config)
        builder = MemoryBuilder(vector_db=vector_db, formation=formation_module, update=update_module, config=config)
        collector = get_collector(config.traj_dir)
        globals()['builder'] = builder
        globals()['retrieval_module'] = retrieval_module
        globals()['collector'] = collector
        globals()['main_config'] = config
        for i, sample in enumerate(data):
            process_sample_logic(i, sample)

    print("Finished processing all samples.")
