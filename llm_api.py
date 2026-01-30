
import os
import json
import time
import random
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI, APIError
from llm_stats import count_and_record, update_stats
from transformers import StoppingCriteriaList, StopStringCriteria
from vllm import LLM, SamplingParams

load_dotenv(verbose=True)

class LLMAPIClientBase(ABC):
    @abstractmethod
    def get_completion(self, 
                       prompt_or_messages: Union[str, List[Dict]], 
                       system_prompt: Optional[str] = None,
                       json_mode: bool = False,
                       stop: List[str] = [],
                       max_retries: int = 3) -> str:
        pass

class OpenAIAPIClient(LLMAPIClientBase):
    def __init__(self, model: str, api_key: str = None, base_url: str = None):
        if OpenAI is None:
            raise ImportError("Please install openai package: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") if "Qwen" not in model else os.environ.get("SILICON_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") if "Qwen" not in model else os.environ.get("SILICON_BASE_URL")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model

    def _construct_messages(self, prompt_or_messages: Union[str, List[Dict]], system_prompt: Optional[str]) -> List[Dict]:
        messages = []

        
        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
            
            if system_prompt and not any(m.get('role') == 'system' for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

        
        elif isinstance(prompt_or_messages, str):
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_or_messages}
                ]
            else:
                messages = [{"role": "user", "content": prompt_or_messages}]

        return messages

    def get_completion(self, 
                       prompt_or_messages: Union[str, List[Dict]], 
                       system_prompt: Optional[str] = None,
                       json_mode: bool = False, 
                       stop: List[str] = [],
                       max_retries: int = 10) -> str:

        messages = self._construct_messages(prompt_or_messages, system_prompt)
        response_format = {"type": "json_object"} if json_mode else {"type": "text"}
        
        for m in messages:
            if m['role'] == 'user':
                pass
            elif m['role'] == 'system':
                pass
            elif m['role'] == 'assistant':
                pass

        
        for attempt in range(max_retries + 1):
            try:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format=response_format,
                        stop=stop if stop and "gpt-5" not in self.model else None, 
                        temperature=0.7 
                    )
                    
                except Exception as e:
                    error_str = str(e).lower()
                    if stop and ("stop" in error_str or "argument" in error_str or "parameter" in error_str):
                        logger.warning(f"Model {self.model} may not support 'stop' tokens. Retrying without stop. Error: {e}")
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            response_format=response_format,
                            temperature=0.7
                        )
                    else:
                        raise e 

                content = response.choices[0].message.content
                if stop:
                    for s_token in stop:
                        if s_token in content:
                            content = content.split(s_token)[0]

                try:
                    count_and_record(messages, content, self.model)
                except Exception as e:
                    logger.warning(f"Failed to record token usage: {e}")

                
                return content

            except Exception as e:
                logger.error(f"[OpenAI API Error] Attempt {attempt}/{max_retries}: {e}")

                if attempt < max_retries:
                    
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached. Returning empty string.")
                    return ""

        return ""

class LocalClient(LLMAPIClientBase):
    def __init__(self, model_name_or_path: str, device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("Please install torch and transformers: pip install torch transformers accelerate")

        self.model_name = model_name_or_path
        logger.info(f"Loading local model from {model_name_or_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                device_map=device, 
                torch_dtype=torch.bfloat16,  
                trust_remote_code=True
            )
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name_or_path}: {e}")
            raise e

    def _construct_messages(self, prompt_or_messages: Union[str, List[Dict]], system_prompt: Optional[str]) -> List[Dict]:
        messages = []

        
        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
            
            if system_prompt and not any(m.get('role') == 'system' for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

        
        elif isinstance(prompt_or_messages, str):
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_or_messages}
                ]
            else:
                messages = [{"role": "user", "content": prompt_or_messages}]

        return messages

    def get_completion(self, 
                       prompt_or_messages: Union[str, List[Dict]], 
                       system_prompt: Optional[str] = None, 
                       json_mode: bool = False,
                       stop: List[str] = [],
                       max_tries:int = 3,
                       if_think:bool = False) -> str:

        
        messages = self._construct_messages(prompt_or_messages, system_prompt)
        for m in messages:
            if m['role'] == 'user':
                
                pass
            elif m['role'] == 'system':
                
                pass
            elif m['role'] == 'assistant':
                
                pass

        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=if_think
            )
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            
            text = messages[-1]['content']
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        input_token_count = model_inputs.input_ids.shape[1]

        stopping_criteria = None
        if stop:
            criteria = StopStringCriteria(tokenizer=self.tokenizer, stop_strings=stop)
            stopping_criteria = StoppingCriteriaList([criteria])
        
        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=65536, 
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            
            output_token_count = len(output_ids)
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

            
            if stop:
                for s_token in stop:
                    if s_token in response:
                        response = response.split(s_token)[0]

            
            update_stats(input_token_count, output_token_count, self.model_name, 0.0, is_external_api=False)
            
            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

class VLLMClient(LLMAPIClientBase):
    def __init__(self, model_name_or_path: str, device: str = "cuda:0", tensor_parallel_size: int = 1):

        self.model_name = model_name_or_path
        self.device = device

        if "cuda:" in device:
            gpu_id = device.split(":")[-1]
        else:
            gpu_id = "0"

        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            logger.info(f"Loading VLLM model from {model_name_or_path} on GPU {gpu_id}...")

            self.llm = LLM(
                model=model_name_or_path,
                trust_remote_code=True,
                tensor_parallel_size=tensor_parallel_size,
                dtype="bfloat16",
                gpu_memory_utilization=0.8,
                max_model_len=65536,
            )
            self.tokenizer = self.llm.get_tokenizer()
            logger.info(f"VLLM model loaded successfully on GPU {gpu_id}.")

        except Exception as e:
            logger.error(f"Failed to load VLLM model {model_name_or_path}: {e}")
            raise e
        finally:
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def _construct_messages(self, prompt_or_messages: Union[str, List[Dict]], system_prompt: Optional[str]) -> List[Dict]:
        messages = []

        if isinstance(prompt_or_messages, list):
            messages = prompt_or_messages
            if system_prompt and not any(m.get('role') == 'system' for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
        elif isinstance(prompt_or_messages, str):
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_or_messages}
                ]
            else:
                messages = [{"role": "user", "content": prompt_or_messages}]

        return messages

    def get_completion(self, 
                       prompt_or_messages: Union[str, List[Dict]], 
                       system_prompt: Optional[str] = None,
                       json_mode: bool = False,
                       stop: List[str] = [],
                       max_retries: int = 3) -> str:
        
        messages = self._construct_messages(prompt_or_messages, system_prompt)

        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except TypeError:
            
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using simple concatenation.")
            prompt_text = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt_text += f"System: {content}\n\n"
                elif role == 'user':
                    prompt_text += f"User: {content}\n\n"
                elif role == 'assistant':
                    prompt_text += f"Assistant: {content}\n\n"
            prompt_text += "Assistant: "

        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=1024,
            truncate_prompt_tokens = -1,
            min_p = 0.05,
            stop=stop if stop else None,
        )

        for attempt in range(max_retries):
            try:
                outputs = self.llm.generate([prompt_text], sampling_params)
                response = outputs[0].outputs[0].text
                if stop:
                    for s_token in stop:
                        if s_token in response:
                            response = response.split(s_token)[0]

                try:
                    input_tokens = len(prompt_text.split()) * 1.3
                    output_tokens = len(response.split()) * 1.3
                    update_stats(int(input_tokens), int(output_tokens), self.model_name, 0.0, is_external_api=False)
                except Exception as e:
                    logger.warning(f"Failed to record token usage: {e}")

                return response

            except Exception as e:
                logger.error(f"[VLLM Error] Attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached. Returning empty string.")
                    return ""