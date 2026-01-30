import threading
import tiktoken
from typing import Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger



OPENAI_PRICING = {
    "gpt-5": {"input": 0.00125, "output": 0.010},
    "gpt-5-2025-08-07": {"input": 0.00125, "output": 0.010},
    "gpt-5-turbo": {"input": 0.00125, "output": 0.010},
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
    "gpt-5-mini-2025-08-07": {"input": 0.00025, "output": 0.002},
    "gpt-5-pro": {"input": 0.015, "output": 0.120},
    "o3": {"input": 0.002, "output": 0.008},
    "o3-2025-06-10": {"input": 0.002, "output": 0.008},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o3-pro": {"input": 0.020, "output": 0.080},
    "gpt-4.5-preview": {"input": 0.075, "output": 0.150},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-2024-08-06": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "o1": {"input": 0.015, "output": 0.060},
    "o1-preview": {"input": 0.015, "output": 0.060},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
}

QWEN_PRICING = {
    "Qwen3-14B":{"input":0.07/1000, "output":0.21/1000}
}


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    model_lower = model.lower()
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        
        if any(x in model_lower for x in ["gpt-5", "o3", "o1", "gpt-4o", "chatgpt-4o"]):
            return tiktoken.get_encoding("o200k_base")
        
        elif any(x in model_lower for x in ["gpt-4", "gpt-3.5"]):
            return tiktoken.get_encoding("cl100k_base")
        else:    
            return None
            


def num_tokens_from_messages(messages: List[Dict], model: str) -> int:
    encoding = get_encoding_for_model(model)
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            
            
            if key == "content" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            if encoding is not None:
                                num_tokens += len(encoding.encode(item["text"]))
                            else:
                                num_tokens += 0
                        
            elif isinstance(value, str):
                if encoding is not None:
                    num_tokens += len(encoding.encode(value))
                else:
                    num_tokens += 0
                
            if key == "name":
                num_tokens += tokens_per_name
                
    num_tokens += 3  
    return num_tokens


def num_tokens_from_string(string: str, model: str) -> int:
    if not string:
        return 0
    encoding = get_encoding_for_model(model)
    if encoding is not None:
        return len(encoding.encode(string))
    else:
        return 0



@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class APIStats:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    total_cost: float = 0.0
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    tokens_by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    costs_by_model: Dict[str, float] = field(default_factory=dict)
    call_history: list = field(default_factory=list)


class LLMStatsCollector:
    """
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LLMStatsCollector, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._stats_lock = threading.Lock()
        self.stats = APIStats()
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = None
        
        
        if model in OPENAI_PRICING:
            pricing = OPENAI_PRICING[model]
        
        
        if pricing is None:
            
            sorted_keys = sorted(OPENAI_PRICING.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if model.startswith(key):
                    pricing = OPENAI_PRICING[key]
                    break
        
        
        if pricing is None:
            if "gpt-5" in model:
                if "mini" in model:
                    pricing = OPENAI_PRICING["gpt-5-mini"]
                elif "pro" in model:
                    pricing = OPENAI_PRICING["gpt-5-pro"]
                else:
                    pricing = OPENAI_PRICING["gpt-5"]
            elif "o3" in model:
                if "mini" in model:
                    pricing = OPENAI_PRICING["o3-mini"]
                else:
                    pricing = OPENAI_PRICING["o3"]
            elif "gpt-4o" in model:
                if "mini" in model:
                    pricing = OPENAI_PRICING["gpt-4o-mini"]
                else:
                    pricing = OPENAI_PRICING["gpt-4o"]
            elif "Qwen3-14B" in model:
                pricing = QWEN_PRICING["Qwen3-14B"]
            else:
                logger.warning(f"Cost calculation: Unknown model pricing for '{model}', defaulting to 0.")
                return 0.0

        input_cost = (input_tokens / 1000.0) * pricing["input"]
        output_cost = (output_tokens / 1000.0) * pricing["output"]
        return input_cost + output_cost

    def record_call(self, 
                    messages: Union[List[Dict], str], 
                    response_text: str,
                    model: str,
                    is_external_api: bool = True) -> float:
        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        try:
            
            input_tokens = num_tokens_from_messages(messages, model)
            output_tokens = num_tokens_from_string(response_text, model)
        except Exception as e:
            logger.error(f"Token calculation failed: {e}")
            return 0.0

        
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        
        self._update_stats(input_tokens, output_tokens, model, cost, is_external_api)
        
        return cost

    def _update_stats(self, input_tokens, output_tokens, model, cost, is_external_api):
        total_tokens = input_tokens + output_tokens
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model=model
        )

        with self._stats_lock:
            self.stats.total_input_tokens += input_tokens
            self.stats.total_output_tokens += output_tokens
            self.stats.total_tokens += total_tokens
            self.stats.total_calls += 1
            self.stats.total_cost += cost
            
            
            if model not in self.stats.calls_by_model:
                self.stats.calls_by_model[model] = 0
                self.stats.tokens_by_model[model] = {"input": 0, "output": 0, "total": 0}
                self.stats.costs_by_model[model] = 0.0
            
            self.stats.calls_by_model[model] += 1
            self.stats.tokens_by_model[model]["input"] += input_tokens
            self.stats.tokens_by_model[model]["output"] += output_tokens
            self.stats.tokens_by_model[model]["total"] += total_tokens
            self.stats.costs_by_model[model] += cost
            
            
            if len(self.stats.call_history) >= 1000:
                self.stats.call_history.pop(0)
            self.stats.call_history.append({
                "usage": usage,
                "cost": cost,
                "is_external_api": is_external_api
            })
            
            current_total_cost = self.stats.total_cost

        
        logger.info(
            f"[LLM Stats] {model} | In:{input_tokens} Out:{output_tokens} | "
            f"Cost: ${cost:.6f} | Total: ${current_total_cost:.4f} |"
            f"Total Calls: {self.stats.total_calls} | Total Input Tokens: {self.stats.total_input_tokens:,} | Total Output Tokens: {self.stats.total_output_tokens:,}"
        )

    def get_summary(self) -> str:
        with self._stats_lock:
            stats = self.stats
            
        lines = [
            "=" * 60,
            f"📊 LLM API Usage Summary (Date: {datetime.now().strftime('%Y-%m-%d')})",
            "=" * 60,
            f"Total Calls : {stats.total_calls}",
            f"Total Tokens: {stats.total_tokens:,}",
            f"  - Input   : {stats.total_input_tokens:,}",
            f"  - Output  : {stats.total_output_tokens:,}",
            f"Total Cost  : ${stats.total_cost:.6f} (≈ ¥{stats.total_cost * 7.25:.2f})",
            "-" * 60,
            f"{'Model':<25} | {'Calls':<6} | {'Cost ($)':<12} | {'Tokens':<10}",
            "-" * 60,
        ]
        
        for model in sorted(stats.costs_by_model.keys()):
            cost = stats.costs_by_model[model]
            calls = stats.calls_by_model[model]
            tokens = stats.tokens_by_model[model]['total']
            lines.append(f"{model:<25} | {calls:<6} | {cost:<12.6f} | {tokens:<10,}")
            
        lines.append("=" * 60)
        return "\n".join(lines)






_stats_collector = LLMStatsCollector()

def get_stats_collector() -> LLMStatsCollector:
    return _stats_collector

def count_and_record(messages: Union[List[Dict], str], response: str, model: str) -> float:
    return _stats_collector.record_call(messages, response, model)

def update_stats(input_tokens: int, output_tokens: int, model: str, cost: float, is_external_api: bool = True) -> float:
    return _stats_collector._update_stats(input_tokens, output_tokens, model, cost, is_external_api)

if __name__ == "__main__":
    
    logger.add("llm_cost_2025.log")
    
    
    msgs = [{"role": "user", "content": "Tell me about the future."}]
    resp = "GPT-5 is here."
    count_and_record(msgs, resp, "gpt-5-mini-2025-08-07")       
    
    
    count_and_record(msgs, resp, "gpt-5")
    
    
    count_and_record(msgs, resp, "o3")

    print(get_stats_collector().get_summary())