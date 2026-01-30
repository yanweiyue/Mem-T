# Mem-T: Memory-Augmented Tree Search for Long-Term Memory

## 🤔 Why Mem-T?

Think of a student taking a long, difficult exam.

* **Previous Agents:** They answer 100 questions but only get a single "Pass" or "Fail" grade at the very end. They have no idea which answers were right or wrong.
* **Mem-T:** It gets a checkmark or cross **immediately after every single step**. Because it receives constant, clear feedback (dense rewards), it learns much faster, makes fewer mistakes, and is cheaper to run.

![intro](assets\intro.png)

## 👋🏻 Method Overview

Mem-T operates as an autonomous agent with a **hierarchical memory system** consisting of Working, Factual, Experiential, and Raw memory modules to manage different types of information. Its workflow is optimized through a novel training framework called **MOT-GRPO** (Memory Operation Tree-guided GRPO). Mem-T works like a smart librarian who both organizes books and helps you find them:

* **1. Smart Organizing (Construction):**
  Instead of piling information up randomly, Mem-T sorts incoming data into specific boxes: **Facts** (what happened), **Experiences** (how to solve problems), and **Raw Data** (exact details), **Working Memory** (summary).
* **2. Step-by-Step Hunting (Retrieval):**
  When you ask a question, Mem-T doesn't just guess. It creates a **"Search Tree"** to explore different paths step-by-step. If a path finds good clues, Mem-T gets an immediate reward, teaching it to find the best answers quickly without wasting time.

![pipeline](assets\main.png)

## 🛠️ Installation

1. **Clone the repository**

   ```bash
   git clone git@github.com:yanweiyue/Mem-T.git
   cd MemT
   ```
2. **Install dependencies**

   ```bash
   # Install general requirements
   pip install -r requirements.txt
   ```

## 📂 Directory Structure

```
data
   └── locomo
      └── locomo10.jsonl
   └── hotpotqa
      └── eval_400.jsonl
   └── longmemeval
      └── longmemeval_s_cleaned.json
   └── narrativeqa
      └── test.parquet
database
   └── locomo
   └── ...
logs
   └── ...
models
   └── EdwinYue/Mem-T-4B
traj
   └── xxxx
      └── mem_trajectories.jsonl
      └── qa_trajectories.jsonl
      └── res.jsonl
```

- `data/`: Benchmark datasets.
- `database/`: Pre-constructed/Constructed memory databases.
- `models/`: Model checkpoints.
- `traj/`: Execution trajectories and reasoning traces.
- `logs/`: Runtime logs.

## 🤖 Models

Our tuned model checkpoint is available on HuggingFace:

- **Mem-T-4B**: [https://huggingface.co/EdwinYue/Mem-T-4B](https://huggingface.co/EdwinYue/Mem-T-4B)
  Please download this model and put it in the `models/` folder.

## 📊 Datasets

We utilize the following benchmarks to evaluate the long-term memory and reasoning capabilities of our model:

### 🧩 Locomo [In Domain]

A long-context reasoning benchmark designed to test the model's ability to retrieve and synthesize information over extended contexts.

Put this memory store in `data/locomo/`.

- **Download**: [Link](https://drive.google.com/file/d/1jlAr2x8uxjOy_dx-2A1Jkz_tHokGf2cn/view?usp=sharing)

### ❓ HotpotQA-56K [OOD]

A large-scale dataset focusing on multi-hop question answering, requiring the agent to perform multiple reasoning steps to derive the correct answer.

Put this memory store in `data/hotpot/`.

- **Download**: [Link](https://drive.google.com/file/d/1qbdM4eje-OJ_3aLrDrYE_sOl87w0WDcG/view?usp=sharing)

## 🧠 Memory Database [Recommend]

Pre-constructed **Locomo Memory Store**: [Download Link](https://drive.google.com/file/d/1ZyJmSni1I62p0pNLJ79NWIze8a8h2SNG/view?usp=sharing)

Put this memory store in `database/`.

Please ensure these files are correctly placed and referenced in `config.py` before starting the ChromaDB server.

## 📈 Trajectory Examples [Optional]

Analyze the reasoning process of our Memory Agent during inference and review the Locomo results reported in our paper:
[Download Trajectory Data](https://drive.google.com/file/d/1HaaMaMv_JmIF1FUCqBVccBBkT3zgUMjm/view?usp=sharing)
This is only intended to help you better understand the algorithmic details and directly verify the results reported in the paper; it is entirely optional.

## ⚡ Quick Start (Inference & Evaluation)

### 1. Start the Vector Database

Before initiating any memory operations, please launch the ChromaDB server:

```bash
sh start_chromadb_server.sh
```

*Note: If you have downloaded our pre-constructed memory bank, ensure the configuration path points to your local directory.*

### 2. Run Evaluation

To evaluate the model on a specific dataset (configured in `config.py` or via command-line arguments):

```bash
# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Run the main evaluation script
python main.py
```

### 3. LLM-as-a-Judge Evaluation

After generating results, employ the LLM judge to assess the quality of the responses:

```bash
python llm_judge.py --input <path_to_qa_trajectories> --output <path_to_result> --benchmark <benchmark_name> --metrics-only
```

Without the `--metrics-only` flag, an LLM-as-a-judge evaluation will be performed. Before proceeding, please ensure your OpenAI Base URL and API Key are configured in the `.env` file.

## ⚙️ Configuration

Key configurations are managed in `config.py`. Customizable parameters include:

- `USE_LOCAL_LLM`: Toggle between local vLLM deployment and OpenAI API.
- `vector_db`: Configuration for ChromaDB (host, port, persistence settings).
- `data_name`: Target dataset for evaluation or training.

The training script will be coming soon.

---

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:

## 🙏 Acknowledgements

We express our gratitude to the following repositories for their valuable code and datasets:

- **Lightmem** and **GAM**: For their excellent memory agent implementations and LLM-as-a-judge prompt designs.
- **verl**, **Search-R1**, and **Tree-GRPO**: For their robust RL implementation frameworks.
- **MemAgent** and **CompassMem**: For their hard-worked dataset processing.
