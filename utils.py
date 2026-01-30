import random
import os
import numpy as np
import torch
from typing import Tuple

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Global seed set to {seed}")

def format_memory_content(content: str, source: str) -> str:
    if not source:
        return content
    return f"{content}\n\n----- ORIGINAL SOURCE -----\n{source}"

def parse_memory_content(document: str) -> Tuple[str, str]:
    separator = "\n\n----- ORIGINAL SOURCE -----\n"
    if separator in document:
        parts = document.split(separator)
        
        return parts[0], separator.join(parts[1:]) 
    else:
        return document, ""
