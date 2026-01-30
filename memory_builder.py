
import json
import re
from typing import List, Dict, Any
from vector_db import VectorDBBase
from memory_formation import MemoryFormation
from memory_update import MemoryUpdate
from config import SystemConfig
from loguru import logger
import time
from trajectory_logger import get_collector, MemTrajectoryLog
import uuid
from utils import format_memory_content, parse_memory_content
from tqdm import tqdm


class MemoryBuilder:
    def __init__(self, 
                 vector_db: VectorDBBase, 
                 formation: MemoryFormation,
                 update: MemoryUpdate,
                 config: SystemConfig):
        self.vector_db = vector_db
        self.formation = formation 
        self.update = update       
        self.config = config
        self.log_filepath = config.log_path
        self.from_scratch = config.vector_db.from_scratch
        self.k_turns = config.memory.summary_context_turns
        self.update_retrieval_topk = config.memory.update_retrieval_topk
        self.benchmark_name = config.data_name if hasattr(config, "data_name") else "longmemeval"

        self.collector = get_collector(config.traj_dir)
        self.BASE_C_TURNS = "turns"
        self.BASE_C_FACTS = "facts"
        self.BASE_C_EXPERIENCES = "experiences"
        self.BASE_C_PERSONAS = "personas"
        self.BASE_C_SUMMARY = "summary"

    def _init_sample_collections(self, sample_id: str):
        for base in [self.BASE_C_TURNS, self.BASE_C_FACTS, self.BASE_C_EXPERIENCES, self.BASE_C_PERSONAS, self.BASE_C_SUMMARY]:
             name = f"{sample_id}_{base}"
             if self.from_scratch:
                self.vector_db.delete_collection(name)
             self.vector_db.create_collection(name, get_or_create=True)

    def build_from_sample(self, sample: Dict[str, Any]):
        
        sample_id = sample.get("conversation", [{}])[0].get("metadata",{}).get("sample_id","")
        if not sample_id:
            sample_id = "unknown_sample"
            logger.warning("Sample missing sample_id, using 'unknown_sample' as default.")

        
        self._init_sample_collections(sample_id)
        
        sessions = sample.get("conversation", [])
        
        total_batches = 0
        for session in sessions:
            turns_count = len(session.get("session_turns", []))
            
            if turns_count > 0:
                batches_in_session = (turns_count + self.k_turns - 1) // self.k_turns
                total_batches += batches_in_session

        
        logger.info(f"Start processing sample {sample_id} with {total_batches} batches...")
        with tqdm(total=total_batches, desc=f"Sample {sample_id}", unit="batch") as pbar:
            for session in sessions:
                self._process_session(session, sample_id, pbar)

    def _process_session(self, session: Dict, sample_id: str, pbar: tqdm = None):
        session_id = session["session_id"]
        turns = session["session_turns"]
        metadata = session.get("metadata", {})
        turn_datetime = metadata.get("session_time", "")
        speaker_a = metadata.get("speaker_a",f"speakerA_{session_id}")
        speaker_b = metadata.get("speaker_b",f"speakerB_{session_id}")
        
        if pbar:
            pbar.set_postfix({"session": session_id})
        
        c_turns = f"{sample_id}_{self.BASE_C_TURNS}"
        c_facts = f"{sample_id}_{self.BASE_C_FACTS}"
        c_exp = f"{sample_id}_{self.BASE_C_EXPERIENCES}"
        c_personas = f"{sample_id}_{self.BASE_C_PERSONAS}"
        c_summary = f"{sample_id}_{self.BASE_C_SUMMARY}"
        prev_summary = ""
        prev_summary_source = ""
        prev_summary_turns = [] 
        prev_personas_map = {}
        prev_personas_source_map = {}
        prev_personas_turns_map = {} 
        summary_src_turn_ids = []
        personas_src_turn_ids = {}
        
        summary_op_ids = []
        personas_op_ids = {}

        
        for speaker in [speaker_a, speaker_b]:
            if not speaker:
                continue
            res = self.vector_db.get(c_personas, ids=[speaker], include=["documents", "metadatas"]) 
            docs = res.get("documents", [])
            metadatas = res.get("metadatas", [])
            if docs and docs[0]:
                content, source = parse_memory_content(docs[0])
                prev_personas_map[speaker] = content
                prev_personas_source_map[speaker] = source
            else:
                prev_personas_map[speaker] = ""
                prev_personas_source_map[speaker] = ""
                
            if metadatas and metadatas[0]:
                personas_src_turn_ids[speaker] = metadatas[0].get("source_turn_ids", [])
                personas_op_ids[speaker] = metadatas[0].get("op_ids", [])
                existing_turns = metadatas[0].get("original_turns", [])
                
                if isinstance(existing_turns, str):
                    try:
                        existing_turns = json.loads(existing_turns)
                    except:
                        existing_turns = [existing_turns]
                prev_personas_turns_map[speaker] = existing_turns
            else:
                personas_src_turn_ids[speaker] = []
                personas_op_ids[speaker] = []
                prev_personas_turns_map[speaker] = []

        
        for i in range(0, len(turns), self.k_turns):
            batch = turns[i:i+self.k_turns]
            
            batch_turn_ids = [t["turn_id"] for t in batch]

            current_text = ""
            current_turns_list = []
            for j, turn in enumerate(batch):
                turn_text = turn.get("text", "")
                turn_speaker = turn.get("speaker", "Unknown")
                turn_speak_to = speaker_b if turn_speaker == speaker_a else speaker_a
                formatted_turn = f"{turn_speaker} speak to {turn_speak_to} at {turn_datetime}: {turn_text}"
                current_text += formatted_turn + "\n"
                current_turns_list.append(formatted_turn)
            
            
            batch_id = f"{session_id}_batch_{i // self.k_turns}"
            self.vector_db.add(c_turns, ids=[batch_id], documents=[current_text], 
                               metadatas=[{"id":batch_id, "col_name":c_turns, "session_id": session_id, "turn_time": turn_datetime, "source_turn_ids": [batch_turn_ids]}])

            
            personas_text_list = [f"Name: {k}, Profile: {v}" for k, v in prev_personas_map.items()]
            prev_personas_str = "\n".join(personas_text_list)
            
            formation_messages = self.formation.construct_prompt(current_text, prev_summary, prev_personas_str)
            formation_response = self.formation.llm_executor.get_completion(formation_messages)
            
            formation_op_id = str(uuid.uuid4())
            formation_mem_ids = []
            formation_col_names = []

            
            tool_calls = self._parse_tool_calls(formation_response)
            extracted_items = []
            
            for tc in tool_calls:
                
                result_obj = self.formation.execute_tool(tc['name'], tc['arguments'])
                if result_obj:
                    
                    if result_obj['type'] == 'summary':
                        summary_src_turn_ids.append(batch_turn_ids) 
                        summary_op_ids.append({"formation": formation_op_id})
                        prev_summary_turns.extend(current_turns_list)
                        
                        
                        new_source = "\n".join(prev_summary_turns)
                        full_document = format_memory_content(result_obj['document'], new_source)
                        
                        self.vector_db.upsert(c_summary, ids=[session_id], documents=[full_document], 
                                              metadatas=[{"id":session_id, "col_name":c_summary,"session_id": session_id, "turn_time": turn_datetime, 
                                                          "source_turn_ids": summary_src_turn_ids,"op_ids": summary_op_ids,"updated_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                          "original_turns": prev_summary_turns, "memory_content": result_obj['document']}])
                        prev_summary = result_obj['document']
                        prev_summary_source = new_source
                        
                        formation_mem_ids.append(session_id)
                        formation_col_names.append(c_summary)
                    elif result_obj['type'] == 'persona':
                        name = result_obj['name']
                        personas_src_turn_ids.get(name, []).append(batch_turn_ids) 
                        personas_op_ids.get(name, []).append({"formation": formation_op_id})
                        
                        current_persona_turns = prev_personas_turns_map.get(name, [])
                        current_persona_turns.extend(current_turns_list)
                        prev_personas_turns_map[name] = current_persona_turns
                        
                        
                        new_source = "\n".join(current_persona_turns)
                        full_document = format_memory_content(result_obj['document'], new_source)
                        
                        self.vector_db.upsert(c_personas, ids=[name], documents=[full_document], 
                                              metadatas=[{"id":name, "col_name":c_personas,"name": name, "turn_time": turn_datetime, 
                                                          "source_turn_ids": personas_src_turn_ids.get(name, []),"op_ids": personas_op_ids.get(name, []), "updated_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                          "original_turns": current_persona_turns, "memory_content": result_obj['document']}])
                        
                        prev_personas_map[name] = result_obj['document']
                        prev_personas_source_map[name] = new_source
                        
                        formation_mem_ids.append(name)
                        formation_col_names.append(c_personas)
                    else:
                        result_obj["turn_time"] = turn_datetime
                        result_obj["original_text"] = current_text 
                        result_obj["original_turns"] = current_turns_list
                        extracted_items.append(result_obj) 

            
            for item in extracted_items:
                col_name = c_facts if item['type'] == 'fact' else c_exp
                
                
                search_res = self.vector_db.search(col_name, query_texts=[item['document']], top_k=self.update_retrieval_topk, include=["documents", "metadatas"])
                related_items = []
                related_ids = search_res.get('ids', [[]])[0]
                related_docs = search_res.get('documents', [[]])[0]
                related_metas = search_res.get('metadatas', [[]])[0]

                for rid, rdoc, rmeta in zip(related_ids, related_docs, related_metas):
                    if rmeta is None: rmeta = {}
                    r_content, r_source = parse_memory_content(rdoc)
                    
                    related_items.append({
                        "id": rid, 
                        "document": r_content, 
                        "turn_time": rmeta.get('turn_time'),
                        "start_time": rmeta.get('start_time'), 
                        "end_time": rmeta.get('end_time'),
                    })

                
                schemas = self.update.get_tool_schemas(col_name)
                potential_item = {
                    "type": item['type'],
                    "document": item['document'],
                    "turn_time": item['turn_time'],
                    "start_time": item['start_time'] if 'start_time' in item else "",
                    "end_time": item['end_time'] if 'end_time' in item else "",
                }
                messages = self.update.construct_prompt(potential_item, related_items, schemas)
                response = self.update.llm_executor.get_completion(messages)

                update_op_id = str(uuid.uuid4())     
                tool_calls = self._parse_tool_calls(response)
                update_mem_ids = []
                update_col_names = []
                for tc in tool_calls:
                    tc['arguments'].update({"source_turn_ids": batch_turn_ids})
                    tc['arguments'].update({"op_ids": {"formation": formation_op_id, "update": update_op_id}})
                    tc['arguments'].update({"original_text": current_text})
                    tc['arguments'].update({"original_turns": current_turns_list})
                    
                    update_feedback = self.update.execute_tool(col_name, tc['name'], tc['arguments'])
                    if not update_feedback or ("status" in update_feedback and update_feedback["status"] == "failed"):
                        logger.warning(f"Update tool {tc['name']} returned failed status result.")
                        continue
                    update_mem_ids.append(update_feedback["id"])
                    update_col_names.append(col_name)
                    formation_mem_ids.append(update_feedback["id"])
                    formation_col_names.append(col_name)
                
                
                self.collector.log_mem_step(MemTrajectoryLog(
                    sample_id=sample_id,
                    phase="update",
                    session_id=session_id,
                    benchmark_name=self.benchmark_name,
                    op_id=update_op_id,
                    input_context=messages,
                    llm_response=response,
                    parsed_output=self._parse_tool_calls(response),
                    source_turn_ids=batch_turn_ids,
                    metadata={"target_collections": update_col_names, "memory_ids": update_mem_ids}
                ))

            self.collector.log_mem_step(MemTrajectoryLog(
                sample_id = sample_id,
                phase = "formation",
                session_id = session_id,
                benchmark_name = self.benchmark_name,
                op_id = formation_op_id,
                input_context = formation_messages,
                llm_response = formation_response,
                parsed_output = self._parse_tool_calls(formation_response),
                source_turn_ids = batch_turn_ids,
                metadata={"target_collections": formation_col_names, "memory_ids": formation_mem_ids}
            ))

            if pbar:
                pbar.update(1)

    def _parse_tool_calls(self, llm_output: str) -> List[Dict]:
        tool_calls = []
        if not llm_output:
            return []
        if llm_output.startswith("<tool_call>") and llm_output.endswith("}"):
            llm_output = f"{llm_output}</tool_call>"

        patterns = re.findall(r"<tool_call>(.*?)</tool_call>", llm_output, re.DOTALL)
        
        if not patterns:
            if "{" in llm_output: 
                logger.warning(f"LLM output contains JSON but missing <tool_call> tags: {llm_output[:100]}...")
            return []

        for content in patterns:
            content = content.strip()
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    for d in data:
                        if isinstance(d, dict):
                            if "name" in d and "arguments" in d and isinstance(d["arguments"], dict):
                                tool_calls.append(d)
                            else:
                                logger.warning(f"Invalid tool call: {d}")
                        else:
                            logger.warning(f"Invalid tool call: {d}")
                elif isinstance(data, dict):
                    if "name" in data and "arguments" in data and isinstance(data["arguments"], dict):
                        tool_calls.append(data)
                    else:
                        logger.warning(f"Invalid tool call: {data}")
            except json.JSONDecodeError:
                logger.warning(f"JSON parse failed inside tool_call: {content}")
        return tool_calls