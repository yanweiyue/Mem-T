
import json
import os
import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class MemTrajectoryLog:
    sample_id: str
    phase: str 
    op_id: str 
    input_context: Any 
    llm_response: str 
    session_id: str =""
    benchmark_name: str = "" 
    parsed_output: Any = ""
    source_turn_ids: List[str] = field(default_factory=list) 
    metadata: Dict[str, Any] = field(default_factory=dict) 

@dataclass
class QATrajectoryLog:
    sample_id: str
    qa_id: str 
    question: Any 
    pred: str 
    gold: str 
    category: str 
    evidence: List[str] = field(default_factory=list) 
    traces: List[Dict] = field(default_factory=list)

class TrainingDataCollector:
    def __init__(self, log_dir: str = "./traj"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.mem_traj_file = os.path.join(log_dir, "mem_trajectories.jsonl")
        self.qa_traj_file = os.path.join(log_dir, "qa_trajectories.jsonl")
        self.lock = threading.Lock()


    def log_mem_step(self, log: MemTrajectoryLog):
        entry = {
            "sample_id": log.sample_id,
            "session_id": log.session_id,
            "benchmark_name": log.benchmark_name,
            "phase": log.phase,
            "op_id": log.op_id,
            "input": log.input_context,
            "output_raw": log.llm_response,
            "output_parsed": log.parsed_output,
            "source_turn_ids": log.source_turn_ids,
            "metadata": log.metadata, 
            "timestamp": time.time()
        }
        with self.lock:
            with open(self.mem_traj_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_qa_step(self, log: QATrajectoryLog):
        entry = {
            "sample_id": log.sample_id,
            "qa_id": log.qa_id,
            "question": log.question,
            "pred": log.pred,
            "gold": log.gold,
            "evidence": log.evidence,
            "traces": log.traces,
            "category": log.category,
            "timestamp": time.time()
        }
        with self.lock:
            with open(self.qa_traj_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    

_collector = None

def get_collector(log_dir: str = "./training_data"):
    global _collector
    if _collector is None:
        _collector = TrainingDataCollector(log_dir)
    return _collector