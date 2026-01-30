import json
import os
import re
import argparse
import threading
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv


from llm_api import OpenAIAPIClient, LocalClient

load_dotenv()


LOCOMO_SYSTEM_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. 
You will be given the following data: 
(1) a question (posed by one user to another user), 
(2) a 'gold' (ground truth) answer, 
(3) a generated answer.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. 
The gold answer will usually be a concise and short answer that includes the referenced topic.

Grading Rules:
1. Be generous. As long as the generated answer touches on the same topic as the gold answer, it should be counted as CORRECT.
2. For time-related questions, if it refers to the same time period as the gold answer (e.g., "last Tuesday" vs "May 7th"), mark it CORRECT.
3. Ignore minor formatting differences.

Output Format:
Return the label in JSON format with the keys "reasoning" and "label".
Do NOT output anything else outside the JSON object.
Example: {"reasoning": "The answer identifies the correct item mentioned in the gold answer.", "label": "CORRECT"}
"""

LOCOMO_USER_PROMPT_TEMPLATE = """
Question: {question} 
Gold answer: {gold_answer} 
Generated answer: {generated_answer} 

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
"""

LONGMEMEVAL_SYSTEM_PROMPT = """Your task is to evaluate whether a model's response correctly answers a given question.
You will be provided with:
(1) A question
(2) A correct answer or rubric
(3) A model's response

Based on the evaluation criteria provided in the user prompt, determine if the model's response is correct.

Output Format:
Return your judgment in JSON format with the keys "reasoning" and "label".
The "label" should be either "CORRECT" or "WRONG".
Example: {"reasoning": "The response contains the correct answer and addresses all key points.", "label": "CORRECT"}
"""

def get_longmemeval_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> tuple:
    
    
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no.\n\n"
                "Question: {}\n\n"
                "Correct Answer: {}\n\n"
                "Model Response: {}\n\n"
                "Is the model response correct? Provide your reasoning and answer with CORRECT or WRONG in JSON format."
            )
            user_prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps "
                "to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors "
                "(e.g., predicting 19 days when the answer is 18), the model's response is still correct.\n\n"
                "Question: {}\n\n"
                "Correct Answer: {}\n\n"
                "Model Response: {}\n\n"
                "Is the model response correct? Provide your reasoning and answer with CORRECT or WRONG in JSON format."
            )
            user_prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, "
                "the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\n"
                "Correct Answer: {}\n\n"
                "Model Response: {}\n\n"
                "Is the model response correct? Provide your reasoning and answer with CORRECT or WRONG in JSON format."
            )
            user_prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\n"
                "Rubric: {}\n\n"
                "Model Response: {}\n\n"
                "Is the model response correct? Provide your reasoning and answer with CORRECT or WRONG in JSON format."
            )
            user_prompt = template.format(question, answer, response)
        else:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
                "Question: {}\n\n"
                "Correct Answer: {}\n\n"
                "Model Response: {}\n\n"
                "Is the model response correct? Provide your reasoning and answer with CORRECT or WRONG in JSON format."
            )
            user_prompt = template.format(question, answer, response)
    else:
        template = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Please answer yes if the model correctly identifies the question as unanswerable. "
            "The model could say that the information is incomplete, or some other information is given "
            "but the asked information is not.\n\n"
            "Question: {}\n\n"
            "Explanation: {}\n\n"
            "Model Response: {}\n\n"
            "Does the model correctly identify the question as unanswerable? "
            "Provide your reasoning and answer with CORRECT or WRONG in JSON format."
        )
        user_prompt = template.format(question, answer, response)
    
    return LONGMEMEVAL_SYSTEM_PROMPT, user_prompt

SYSTEM_PROMPT = LOCOMO_SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = LOCOMO_USER_PROMPT_TEMPLATE



def normalize_text(s: str) -> str:
    if s is None: return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks: return 1.0
    if not gtoks or not ptoks: return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0: return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0: return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

def parse_llm_output(text: str) -> dict:
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    try:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match: return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match: return json.loads(match.group(0))
    except Exception: pass
    label = "WRONG"
    if "CORRECT" in text.upper() and "WRONG" not in text.upper(): label = "CORRECT"
    return {"reasoning": text, "label": label, "parse_error": True}


class JudgeWorker:
    def __init__(self, model_name: str, api_key: str = None, base_url: str = None, is_local: bool = False, benchmark_name: str = "locomo", metrics_only: bool = False):
        self.benchmark_name = benchmark_name
        self.metrics_only = metrics_only  
        
        
        if not self.metrics_only:
            if is_local:
                if LocalClient is None:
                    raise ImportError("LocalClient is not imported. Please check dependencies.")
                self.client = LocalClient(model_name_or_path=model_name)
            else:
                if OpenAIAPIClient is None:
                    raise ImportError("OpenAIAPIClient is not imported. Please check dependencies.")
                self.client = OpenAIAPIClient(model=model_name, api_key=api_key, base_url=base_url)
        else:
            self.client = None
        
        self.write_lock = threading.Lock()

    def evaluate_sample(self, sample: dict) -> dict:
        question = sample.get("question", "")
        gold = sample.get("gold", "I don't know.")
        pred = sample.get("pred", "I don't know.")

        try:
            if isinstance(gold, list):
                f1_scores = [f1_score(pred, g) for g in gold]
                bleu1_scores = [bleu1_score(pred, g) for g in gold]
                f1 = max(f1_scores) if f1_scores else 0.0
                bleu1 = max(bleu1_scores) if bleu1_scores else 0.0
                gold_display = " OR ".join([str(g) for g in gold])
            else:
                f1 = f1_score(pred, str(gold))
                bleu1 = bleu1_score(pred, str(gold))
                gold_display = str(gold)
                
            sample["f1_score"] = f1
            sample["bleu1_score"] = bleu1
        except Exception as e:
            sample["f1_score"] = 0.0
            sample["bleu1_score"] = 0.0
            sample["metric_error"] = str(e)
            gold_display = str(gold)

        
        if self.metrics_only:
            
            sample["judge_reasoning"] = "Skipped (Metrics Only Mode)"
            sample["judge_label"] = "SKIPPED"
        else:
            
            if self.benchmark_name == "longmemeval":
                task = sample.get("question_type") or sample.get("category") or "single-session-user"
                abstention = "_abs" in str(sample.get("sample_id", ""))
                system_prompt, user_prompt = get_longmemeval_prompt(
                    task=task, question=question, answer=gold_display, response=pred, abstention=abstention
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": LOCOMO_SYSTEM_PROMPT},
                    {"role": "user", "content": LOCOMO_USER_PROMPT_TEMPLATE.format(
                        question=question, gold_answer=gold_display, generated_answer=pred
                    )}
                ]

            try:
                response_text = self.client.get_completion(
                    prompt_or_messages=messages, json_mode=True, max_retries=3
                )
                judge_result = parse_llm_output(response_text)
                sample["judge_reasoning"] = judge_result.get("reasoning", "")
                sample["judge_label"] = judge_result.get("label", "WRONG")
                if judge_result.get("parse_error"):
                    sample["judge_parse_error"] = True
            except Exception as e:
                sample["judge_error"] = str(e)
                sample["judge_label"] = "ERROR"
        
        return sample

def print_statistics(args):
    category_stats = {}
    total_correct = 0
    total_wrong = 0
    total_skipped = 0 
    total_error = 0
    qa_id_to_category = {}
    if os.path.exists(args.input):
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        qa_id = item.get("qa_id") or item.get("sample_id")
                        category = item.get("category")
                        if qa_id and category is not None:
                            qa_id_to_category[qa_id] = str(category)
                    except: continue

    with open(args.output, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    category = item.get("category") or item.get("question_category")
                    
                    if category is None:
                        qa_id = item.get("qa_id")
                        if qa_id and qa_id in qa_id_to_category:
                            category = qa_id_to_category[qa_id]
                    
                    category = str(category) if category is not None else "unknown"
                    
                    if category not in category_stats:
                        category_stats[category] = {
                            "correct": 0, "wrong": 0, "error": 0, "skipped": 0, "total": 0,
                            "f1_scores": [], "bleu1_scores": []
                        }
                    
                    label = item.get("judge_label", "").upper()
                    category_stats[category]["total"] += 1
                    
                    if "f1_score" in item and item["f1_score"] is not None:
                        category_stats[category]["f1_scores"].append(item["f1_score"])
                    if "bleu1_score" in item and item["bleu1_score"] is not None:
                        category_stats[category]["bleu1_scores"].append(item["bleu1_score"])
                    
                    if label == "CORRECT":
                        category_stats[category]["correct"] += 1
                        total_correct += 1
                    elif label == "WRONG":
                        category_stats[category]["wrong"] += 1
                        total_wrong += 1
                    elif label == "SKIPPED":
                        category_stats[category]["skipped"] += 1
                        total_skipped += 1
                    else:
                        category_stats[category]["error"] += 1
                        total_error += 1
                except Exception: continue

    
    if category_stats:
        def sort_key(x):
            if x == "unknown": return (2, x)
            elif x.isdigit(): return (0, int(x))
            else: return (1, x)
        
        sorted_categories = sorted(category_stats.keys(), key=sort_key)
        
        total_f1_scores = []
        total_bleu1_scores = []
        
        print("\n" + "="*80)
        print(f"{'Category':<20} | {'Total':<6} | {'Acc':<8} | {'Avg F1':<8} | {'Avg BLEU1':<9} | {'State'}")
        print("-" * 80)

        for cat in sorted_categories:
            stats = category_stats[cat]
            total = stats["total"]
            skipped = stats["skipped"]
            correct = stats["correct"]
            
            
            valid_judge_count = total - skipped
            accuracy = (correct / valid_judge_count * 100) if valid_judge_count > 0 else 0.0
            
            f1_scores = stats.get("f1_scores", [])
            bleu1_scores = stats.get("bleu1_scores", [])
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
            avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0
            
            total_f1_scores.extend(f1_scores)
            total_bleu1_scores.extend(bleu1_scores)
            
            state_str = "Metrics Only" if skipped == total else "Judged"
            acc_str = f"{accuracy:.2f}%" if valid_judge_count > 0 else "N/A"

            print(f"{cat:<20} | {total:<6} | {acc_str:<8} | {avg_f1:.4f}   | {avg_bleu1:.4f}    | {state_str}")
        
        
        total_all = total_correct + total_wrong + total_error + total_skipped
        valid_all = total_correct + total_wrong + total_error
        
        overall_accuracy = (total_correct / valid_all * 100) if valid_all > 0 else 0.0
        overall_avg_f1 = sum(total_f1_scores) / len(total_f1_scores) if total_f1_scores else 0.0
        overall_avg_bleu1 = sum(total_bleu1_scores) / len(total_bleu1_scores) if total_bleu1_scores else 0.0
        
        print("-" * 80)
        print(f"Overall Statistics:")
        print(f"  Total Samples: {total_all}")
        if args.metrics_only or total_skipped == total_all:
            print(f"  Judge Skipped: {total_skipped}")
        else:
            print(f"  Correct: {total_correct} | Wrong: {total_wrong} | Error: {total_error}")
            print(f"  Overall Accuracy (LLM): {overall_accuracy:.2f}%")
        
        print(f"  Overall Avg F1:         {overall_avg_f1:.4f}")
        print(f"  Overall Avg BLEU-1:     {overall_avg_bleu1:.4f}")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="LLM as a Judge Evaluation Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl file")
    parser.add_argument("--model", type=str, default="gpt-5-mini-2025-08-07", help="Model name to use")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent threads")
    parser.add_argument("--local", action="store_true", help="Use LocalClient instead of OpenAIAPIClient")
    parser.add_argument("--benchmark", type=str, default="locomo", choices=["locomo", "longmemeval", "membench"], 
                        help="Benchmark dataset name")
    parser.add_argument("--metrics-only", action="store_true", help="Only calculate F1 and BLEU scores, skip LLM Judge")

    args = parser.parse_args()

    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    processed_ids = set()
    if os.path.exists(args.output):
        print(f"Output file {args.output} exists. Resuming...")
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        uid = item.get("sample_id") or item.get("qa_id")
                        if uid:
                            processed_ids.add(uid)
                    except: continue
        print_statistics(args) 
    
    tasks = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                uid = item.get("sample_id") or item.get("qa_id")
                if uid and uid not in processed_ids:
                    tasks.append(item)

    if not tasks:
        print("All tasks completed.")
        return 

    print(f"Total tasks to process: {len(tasks)}")
    
    try:
        worker = JudgeWorker(
            model_name=args.model, 
            is_local=args.local,
            benchmark_name=args.benchmark,
            metrics_only=args.metrics_only 
        )
    except Exception as e:
        print(f"Worker Initialization Error: {e}")
        return

    max_workers = args.workers if not args.metrics_only else min(32, len(tasks))

    with open(args.output, "a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {executor.submit(worker.evaluate_sample, sample): sample for sample in tasks}
            pbar = tqdm(total=len(tasks), desc="Processing" if args.metrics_only else "Judging")
            
            for future in as_completed(future_to_sample):
                sample_data = future_to_sample[future]
                try:
                    result_sample = future.result(timeout=60) 
                    f_out.write(json.dumps(result_sample, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    uid = sample_data.get("sample_id") or sample_data.get("qa_id")
                    print(f"\n[ERROR] Task {uid} failed: {e}")
                
                pbar.update(1)

    print(f"\nProcess finished. Results saved to {args.output}")
    print_statistics(args)

if __name__ == "__main__":
    main()