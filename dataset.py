
import json
from typing import List, Dict, Any
import os
from loguru import logger
import re
import numpy as np
from datasets import load_dataset

def load_locomo_dataset(locomo_filepath: str) -> List[Dict[str, Any]]:
    """
    Load Locomo dataset and convert it to 'chat_dataset' format.
    """
    chat_dataset = []
    
    try:
        with open(locomo_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f) 
    except FileNotFoundError:
        logger.error(f"Locomo Not Found: {locomo_filepath}")
        return []
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON: {locomo_filepath}")
        return []

    
    for sample in data:
        sample_id = sample.get("sample_id") 
        locomo_conversation = sample.get("conversation") 
        locomo_qa = sample.get("qa") 
        formatted_conversation = []
        formatted_qa = []
        dia_id_to_turn_id = {}  
        if not locomo_conversation or not isinstance(locomo_conversation, dict):
            continue

        logger.info(f"Processing Sample ID: {sample_id}") 

        speaker_a_name = locomo_conversation.get("speaker_a", "SpeakerA")
        speaker_b_name = locomo_conversation.get("speaker_b", "SpeakerB")
        conversation_id = f"{sample_id}_conv"
        
        
        session_ids = sorted(
            [k for k in locomo_conversation.keys() if re.match(r'session_\d+$', k)],
            key=lambda x: int(x.split('_')[1])
        )
            
        for idx in session_ids: 
            session_id = f"{conversation_id}_{idx}"
            session_turns_data = locomo_conversation[idx] 
            if not isinstance(session_turns_data, list):
                continue
                
            session_datetime = locomo_conversation.get(f"{idx}_date_time", "") 
            formatted_turns = []
            for i, turn in enumerate(session_turns_data):
                dia_id = turn.get("dia_id","") 
                if not dia_id:
                    print(f"Warning: Turn ID, Sample {sample_id}, Conversation {conversation_id}, Session: {session_id}, Turn Index: {i} does not have 'dia_id'. Generating one.")
                    turn_id = f"{session_id}_turn_{i}"
                else:
                    turn_id = f"{session_id}_{dia_id}" 
                    dia_id_to_turn_id[dia_id] = turn_id
                
                formatted_turns.append({
                    "turn_id": turn_id,
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", "")
                })
            
            
            formatted_conversation.append({
                "sample_id": sample_id,
                "session_id": session_id,
                "session_turns": formatted_turns,
                "metadata": {
                    "sample_id": sample_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "session_time": session_datetime,
                    "speaker_a": speaker_a_name,
                    "speaker_b": speaker_b_name
                }
            })
        
        
        if isinstance(locomo_qa, list):
            for qa_item in locomo_qa:
                if qa_item.get("category") == 5:
                    continue
                formatted_qa.append({
                    "sample_id": sample_id,
                    "question": qa_item.get("question"),
                    "answer": qa_item.get("answer", "I don't know."),
                    "evidence": list(map(lambda dia_id: dia_id_to_turn_id[dia_id], qa_item.get("evidence"))) if qa_item.get("evidence") else [],
                    "category": qa_item.get("category")
                })
        
        
        chat_dataset.append({
            "qa": formatted_qa,
            "conversation": formatted_conversation
        })

    logger.info(f"{len(chat_dataset)}  'Samples'")
    return chat_dataset


def load_hotpotqa_dataset(filepath="") -> List[Dict[str, Any]]:
    chat_dataset = []
    
    if not filepath or not os.path.exists(filepath):
        logger.error(f"HotpotQA Not Found: {filepath}")
        return []

    logger.info(f"Loading HotpotQA from {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        return []

    logger.info(f"Found {len(data)} samples")

    for idx, sample in enumerate(data): 
        sample_index = sample.get("index", idx)
        sample_id = str(sample_index)
        question = sample.get("input", "")
        answers = sample.get("answers", [])
        answer_str = answers[0] if answers and isinstance(answers, list) else str(answers)
        raw_context = sample.get("context", "")
        formatted_conversation = []
        conversation_id = f"{sample_id}_conv"
        docs = re.split(r'Document \d+:\n', raw_context)
        docs = [d.strip() for d in docs if d.strip()]
        
        for i, doc_content in enumerate(docs):
            session_id = f"{conversation_id}_s{i}"
            turn_id = f"{session_id}_t0"

            formatted_turns = [{
                "turn_id": turn_id,
                "speaker": "wikipedia",
                "text": doc_content
            }]
            title = doc_content.split('\n')[0] if doc_content else f"Document {i+1}"
            formatted_conversation.append({
                "session_id": session_id,
                "session_turns": formatted_turns,
                "metadata": {
                    "sample_id": sample_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "title": title,
                    "speaker_a": "wikipedia",
                    "speaker_b": "wikipedia" 
                }
            })

        formatted_qa = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer_str,
            "evidence": [], 
            "category": "hotpotqa",
        }
        
        chat_dataset.append({
            "qa": [formatted_qa],
            "conversation": formatted_conversation
        })
        
    logger.info(f"{len(chat_dataset)} samples loaded")
    return chat_dataset

    
def train_valid_test_split(data, seed=42):
    total_len = len(data)
    indices = np.arange(total_len)
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_end = int(total_len * 0.1)
    val_end = int(total_len * 0.2)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_data = [data[i] for i in train_indices]
    valid_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, valid_data, test_data