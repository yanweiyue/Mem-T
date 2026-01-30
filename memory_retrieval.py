
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from vector_db import VectorDBBase
from llm_api import LLMAPIClientBase
from config import SystemConfig
from tools_base import BaseTool
from trajectory_logger import get_collector, MemTrajectoryLog 
import uuid


class SearchSummaryTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase):
        super().__init__(
            name="search_summary",
            description="Retrieve relevant summaries to quickly understand the context background.",
            parameters={"query": {"type": "string", "description": "Query string."}},
            required=["query"]
        )
        self.db = vector_db

    def _format_with_source(self, document: str, metadata: Dict, seen_turns: set = None) -> str:
        content = ""
        source_turns = []
        
        
        if metadata and "memory_content" in metadata:
            content = metadata["memory_content"]
            if "original_turns" in metadata:
                source_turns = metadata["original_turns"]
                if isinstance(source_turns, str):
                    try:
                        source_turns = json.loads(source_turns)
                    except:
                        source_turns = [source_turns]
                elif not isinstance(source_turns, list):
                    source_turns = [str(source_turns)]
        else:
            
            from utils import parse_memory_content
            content, source_text = parse_memory_content(document)
            if source_text:
                source_turns = [source_text] 
        
        
        unique_turns = []
        if seen_turns is not None:
             for t in source_turns:
                if t not in seen_turns:
                    unique_turns.append(t)
                    seen_turns.add(t)
        else:
            seen = set()
            for t in source_turns:
                if t not in seen:
                    unique_turns.append(t)
                    seen.add(t)
        
        
        if unique_turns:
            return f"{content}\n\n----- NEW SOURCE CONTEXT -----\n" + "\n".join(unique_turns)
        else:
            return content

    def __call__(self, query: str = "", sample_id: str = "", seen_turns: set = None, **kwargs):
        
        if not query:
            try:
                
                res = self.db.get(f"{sample_id}_summary", include=["documents", "metadatas"])
                if not res or not res.get('documents') or not sample_id:
                    return "", []
                documents = res.get('documents', []) 
                metadatas = res.get('metadatas', []) 
                
                
                data_pairs = []
                for doc, meta in zip(documents, metadatas):
                    if doc:
                        
                        
                        time_val = ""
                        if meta:
                            time_val = meta.get("updated_time") or ""
                        data_pairs.append((doc, time_val, meta))
                if not data_pairs:
                    return "", []
                
                
                data_pairs.sort(key=lambda x: x[1], reverse=True)
                
                
                doc = data_pairs[0][0]
                metadata = data_pairs[0][2]
                summary = self._format_with_source(doc, metadata, seen_turns)
                
                logger.info(f"Retrieved latest summary")
                return summary, [metadata]

            except Exception as e:
                logger.error(f"Error retrieving latest summary: {e}")
                return "", []
        
        else:
            res = self.db.search(f"{sample_id}_summary", query_texts=[query], top_k=1, include=["documents", "metadatas"]) 
            
            if res and res.get('documents') and res['documents'][0]:
                doc = res['documents'][0][0]
                metadata = res['metadatas'][0][0] if res.get('metadatas') and res['metadatas'][0] else {}
                summary = self._format_with_source(doc, metadata, seen_turns)
                metadatas = [metadata]
            else:
                summary = ""
                metadatas = []
                
            return summary, metadatas

class SearchFactsTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, top_k: int):
        super().__init__(
            name="search_facts",
            description = "Retrieve 'Factual Memory' (Concrete, verifiable statements about WHAT happened).\n"
                "Target two specific types of facts:\n"
                "1. User Factual Memory: Verifiable facts about the user's identity, stable preferences, important events, habits, "
                "historical commitments, and specific constraints.\n"
                "2. Environment Factual Memory: Explicit states of the external world, object properties, "
                "document knowledge, or tool states.\n",
            parameters = {"query": {"type": "string",
                                   "description": "A self-contained, semantically rich search query rewritten from the user's intent.\n"
                                                "Instead of raw questions like 'Does he like it?', use specific declarative queries like "
                                                "'User preference regarding spicy food' or 'Attributes of Object X'."}},
            required=["query"]
        )
        self.db = vector_db
        self.top_k = top_k

    def __call__(self, query: str = "", sample_id: str = "", seen_turns: set = None, **kwargs):
        if not query or not sample_id: 
            logger.warning("SearchFactsTool called with empty query.")
            return "", []
        res = self.db.search(f"{sample_id}_facts", query_texts=[query], top_k=self.top_k, include=["documents", "metadatas"])
        metadatas = res["metadatas"][0] if res and res.get("metadatas") else []
        return self._format_results(res, seen_turns), metadatas

    def _format_results(self, res, seen_turns: set = None):
        if not res or not res.get('documents') or not res['documents'][0]:
            return ""
        formatted_memories = []
        new_source_turns = []
        
        from utils import parse_memory_content
        
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            meta_str = f" (Valid Time: {meta.get('start_time', '')} to {meta.get('end_time', '')}. Extracted at {meta.get('turn_time', '')})" if meta else ""
            
            
            content = ""
            source_turns = []
            
            if meta and "memory_content" in meta:
                content = meta["memory_content"]
                if "original_turns" in meta:
                    source_turns = meta["original_turns"]
                    if isinstance(source_turns, str):
                        try:
                            source_turns = json.loads(source_turns)
                        except:
                            source_turns = [source_turns]
                    elif not isinstance(source_turns, list):
                        source_turns = [str(source_turns)]
            else:
                 content, source_text = parse_memory_content(doc)
                 if source_text:
                     source_turns = [source_text]
            
            
            if seen_turns is not None:
                for t in source_turns:
                    if t not in seen_turns:
                        new_source_turns.append(t)
                        seen_turns.add(t)
            else:
                 new_source_turns.extend(source_turns) 

            
            formatted_memories.append(f"{meta_str} {content}")
            
        result = "Found Memories:\n" + "\n".join(formatted_memories)
        
        if new_source_turns:
             result += "\n\n----- NEW SOURCE CONTEXT -----\n" + "\n".join(new_source_turns)
             
        return result

class SearchExperiencesTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, top_k: int):
        super().__init__(
            name="search_experiences",
            description=(
                "Extract 'Experiential Memory' (Actionable lessons, patterns, or HOW-TO perform a task)\n"
                "This tool captures lessons learned, reasoning patterns, and executable skills.:\n"
                "1. Strategy-based: Reusable heuristics, workflows, or insights derived from reasoning (e.g., 'To solve X, method Y is most efficient').\n"
                "2. Case-based: Key trajectories of Success or Failure that serve as examples (e.g., 'Attempting action A under condition B caused error C').\n"
                "3. Skill-based: Validated code snippets, tool usage protocols, or functions that the agent can execute.\n"
                "Avoid recording raw dialogue history; focus on the distilled 'Lesson' or 'Rule'."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "A self-contained, semantically rich search query rewritten from the user's intent. \n"
                        "Formulate problem-solving queries like 'Standard workflow for analyzing finance reports' "
                        "or 'How to handle TimeoutError in API calls'."
                    )
                }
            },
            required=["query"]
        )
        self.db = vector_db
        self.top_k = top_k

    def __call__(self, query: str = "", sample_id: str = "", seen_turns: set = None, **kwargs):
        if not query or not sample_id:
            logger.warning("SearchExperiencesTool called with empty query.") 
            return "", []
        res = self.db.search(f"{sample_id}_experiences", query_texts=[query], top_k=self.top_k, include=["documents", "metadatas"])
        metadatas = res["metadatas"][0] if res and res.get("metadatas") else []
        return self._format_results(res, seen_turns), metadatas
    
    def _format_results(self, res, seen_turns: set = None):
        if not res or not res.get('documents') or not res['documents'][0]:
            return "No experiences found."
        formatted_memories = []
        new_source_turns = []
        from utils import parse_memory_content

        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            meta_str = f" (Occurred Period: {meta.get('start_time', '')} to {meta.get('end_time', '')}. Extracted at {meta.get('turn_time','')})" if meta else ""
            
             
            content = ""
            source_turns = []
            
            if meta and "memory_content" in meta:
                content = meta["memory_content"]
                if "original_turns" in meta:
                    source_turns = meta["original_turns"]
                    if isinstance(source_turns, str):
                        try:
                            source_turns = json.loads(source_turns)
                        except:
                            source_turns = [source_turns]
                    elif not isinstance(source_turns, list):
                        source_turns = [str(source_turns)]
            else:
                 content, source_text = parse_memory_content(doc)
                 if source_text:
                     source_turns = [source_text]
            
            
            if seen_turns is not None:
                for t in source_turns:
                    if t not in seen_turns:
                        new_source_turns.append(t)
                        seen_turns.add(t)
            else:
                 new_source_turns.extend(source_turns)

            
            formatted_memories.append(f"{meta_str} {content}")
            
        result = "Found Experiences:\n" + "\n".join(formatted_memories)
        
        if new_source_turns:
             result += "\n\n----- NEW SOURCE CONTEXT -----\n" + "\n".join(new_source_turns)
             
        return result

class SearchPersonasTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase):
        super().__init__(
            name="search_personas",
            description="Retrieve character profiles or insights for specific individuals.",
            parameters={
                "name": {"type": "string", "description": "Name of the target individual for exact lookup."},
                "query": {"type": "string", "description": "Query string to find personas by traits; ignored if 'name' is provided."}
            },
            required=["query"] 
        )
        self.db = vector_db

    def __call__(self, query: str = "", name: str = "", sample_id: str = "", seen_turns: set = None, **kwargs):
        from utils import parse_memory_content, format_memory_content
        
        def _process_doc(doc, meta, seen_turns: set = None):
            content = ""
            source_turns = []
            
            if meta and "memory_content" in meta:
                content = meta["memory_content"]
                if "original_turns" in meta:
                    source_turns = meta["original_turns"]
                    if isinstance(source_turns, str):
                        try:
                            source_turns = json.loads(source_turns)
                        except:
                            source_turns = [source_turns]
                    elif not isinstance(source_turns, list):
                        source_turns = [str(source_turns)]
            else:
                 content, source_text = parse_memory_content(doc)
                 if source_text:
                     source_turns = [source_text]
            
            
            unique_turns = []
            if seen_turns is not None:
                for t in source_turns:
                    if t not in seen_turns:
                        unique_turns.append(t)
                        seen_turns.add(t)
            else:
                unique_turns = source_turns
            
            if unique_turns:
                return f"{content}\n\n----- NEW SOURCE CONTEXT -----\n" + "\n".join(unique_turns)
            else:
                return content

        
        if name:
            res = self.db.get(f"{sample_id}_personas", ids=[name], include=["documents", "metadatas"])
            if res and res.get('documents') and res['documents'][0] and res["metadatas"] and res["metadatas"][0]:
                full_doc = _process_doc(res['documents'][0], res["metadatas"][0], seen_turns)
                profile = f"Persona Profile ({name}): {full_doc}"
                metadatas = res["metadatas"]
                return profile, metadatas
            return "", []
        
        elif query:
            res = self.db.search(f"{sample_id}_personas", query_texts=[query], top_k=1, include=["documents", "metadatas"])
            if res and res.get('documents') and res['documents'][0]:
                 full_doc = _process_doc(res['documents'][0][0], res['metadatas'][0][0], seen_turns)
                 profile = full_doc
            else:
                 profile = f"No persona found matching query."
            
            metadatas = res["metadatas"][0] if res and res.get("metadatas") else []
            return profile, metadatas
        logger.warning("SearchPersonasTool called with empty query and name.")
        return "", []

class SearchTurnsTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, top_k: int):
        super().__init__(
            name="search_turns",
            description="Retrieve specific raw dialogue history (Raw Turns). \n"
                        "Use this tool for questions about specific past conversations, verifying exact quotes, or checking 'what was' in detail. \n"
                        "Raw turns provide the most authentic context that summaries or facts might miss.",
            parameters={
                "query": {"type": "string", "description": "Keywords or specific quotes."},
                "top_k": {"type": "integer", "description": "The number of turns to retrieve. Default is 5."}
                },
            required=["query"]
        )
        self.db = vector_db

    def __call__(self, query: str = "", sample_id: str = "", top_k: int = 5, seen_turns: set = None, **kwargs):
        if not query or not sample_id:
            logger.warning("SearchTurnsTool called with empty query or sample_id.")
            return "", []
        if top_k>50:
            top_k = 50
        res = self.db.search(f"{sample_id}_turns", query_texts=[query], top_k=top_k, include=["documents", "metadatas"])
        if not res or not res.get('documents') or not res['documents'][0]:
            logger.warning("SearchTurnsTool called with empty result.")
            return "",[]
        formatted = []
        
        
        new_turns = []
        for doc, meta in zip(res['documents'][0], res['metadatas'][0]):
            if seen_turns is not None:
                if doc not in seen_turns:
                    new_turns.append(doc)
                    seen_turns.add(doc)
            else:
                new_turns.append(doc)
        
        if new_turns:
            return "Relevant Turns:\n" + "\n".join(new_turns), res['metadatas'][0]
        else:
            return "No new relevant turns found (some might have been shown in context).", res['metadatas'][0]

class FinishTool(BaseTool):
    def __init__(self, benchmark_name: str = "locomo", category: str = ""):
        self.benchmark_name = benchmark_name
        self.category = str(category) if category else ""
        
        
        description = "Call this when you are confident that you can give the correct final answer. Or you should continue to retrieve more information."
        
        super().__init__(
            name="finish",
            description=description,
            parameters={"answer": {"type": "string", "description": "The concise answer following the Final Result Format."}},
            required=["answer"]
        )

    def __call__(self, answer: str = "", **kwargs):
        if not answer:
            logger.warning("FinishTool called with empty answer.")
            return "", []
        return answer, []

def get_final_result_format(benchmark_name: str, category: str = "") -> str:
    category = str(category) if category else ""
    if benchmark_name == "locomo":
        if category == "3":
            
            return """The Final Result's Format Must Follow These Rules:
1. Provide a short phrase answer, not a full sentence.
2. The question may require you to analyze and infer the answer from the retrieved information.
3. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic numerals in the answer respectively. Numbers are represented by English words by default., eg. prefer **two** not 2.
4. This is an open-domain problem. **When answering this type of question, you can ignore all other requirements that you must be completely confident before responding.** NEVER answer 'I don't know./None./Unknown'. You can perform reasoning based on the retrieved information and your model knowledge. Uncertain inferences can be expressed using 'likely'.
5. When the answer has multiple phrases, connect them with commas don't use 'and'.
6. Ensure your response aligns directly with the question. For instance, start with 'Yes' or 'No' for binary questions, and do not name a province when asked for a country.
7. If the information is not enough, you MUST NOT answer 'Unknown' or 'I don't know.'. Instead, try searching other databases, use different query words or expand the retrieval top-K. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer and MUST NOT say 'Unknown'."""
        else:
            
            return """The Final Result's Format Must Follow These Rules:
1. For questions requiring a date or time, strictly follow the format '15 July, 2023', 'July, 2023'.
2. Pay special attention to relative times like 'yesterday', 'last week', 'last Friday' in the text:
   + Only for last year/ last month/yesterday, calculate the absolute date, precise to year/month/day respectively, eg. 'July, 2023' or '19 July, 2023'.
   + For last week/weekend/Friday/Saturday, or few days ago etc, use 'the week/weekend/Friday before [certain absolute time]' to express **MUST NOT calculate the exact date**, just use the week/weekend/Friday before the certain absolute time, eg. 'the week/weekend/Friday before 15 July, 2023'/ few days before 15 July, 2023;
3. The answer should be the form of a short phrase (roughly a few words) for the following question, not a full sentence.
4. Use exact wording from the original conversation whenever possible.
5. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic numerals in the answer respectively. If it is a quantity or frequency counted by yourself, default to using English word, eg. prefer **two** not 2.
6. When the answer has multiple phrases, connect them with commas don't use 'and'.
7. Ensure your response aligns directly with the question. For instance, start with 'Yes' or 'No' for binary questions, and do not name a province when asked for a country.
8. If the information is not enough, you MUST NOT answer 'Unknown' or 'I don't know.'. Instead, try searching other databases, use different query words or expand the retrieval top-K. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer and MUST NOT say 'Unknown'."""
    elif benchmark_name == "longmemeval":
        return """The Final Result's Format Must Follow These Rules:
1. Please answer the question based on the retrieve memories.
2. For questions asking for opinions and suggestions, you should respond when you have sufficient memory; you cannot refuse to answer.
3. When answering this type of question, you can ignore all other requirements that you must be completely confident before responding."""
    elif benchmark_name == "hotpotqa" or "narrativeqa":
        return """The Final Result's Format Must Follow These Rules:
1. Please answer the question based on the retrieve memories.
2. The answer should be the form of a short phrase (roughly a few words) for the following question, not a full sentence.
3. Use exact wording from the original conversation whenever possible.
4. Answer with ONLY the final answer string; no extra words. 
5. There's no need to repeat the question; just provide the answer phrase directly.
6. If the information is not enough, you MUST NOT answer 'Unknown' or 'I don't know.'. Instead, try searching other databases, use different query words or expand the retrieval top-K. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer and MUST NOT say 'Unknown'."""
    else:
        return """The Final Result's Format Must Follow These Rules:
1. Please answer the question based on the retrieve memories.
2. For questions asking for opinions and suggestions, you should respond when you have sufficient memory; you cannot refuse to answer."""
class AnswerWithMemoriesTool(BaseTool):
    def __init__(self, llm_api: LLMAPIClientBase, benchmark_name: str = "locomo", category: str = ""):
        super().__init__(
            name="answer_with_memories",
            description="Generate a final answer based on the user query and all retrieved memories.",
            parameters={
                "query": {"type": "string", "description": "The user's original question."},
                "memories": {"type": "string", "description": "All retrieved memories concatenated."}
            },
            required=["query", "memories"]
        )
        self.llm = llm_api
        self.benchmark_name = benchmark_name
        self.category = str(category) if category else ""
        self.format_instruction = get_final_result_format(self.benchmark_name, self.category)

    def __call__(self, query: str, memories: str, **kwargs):
        if not query or not memories:
             logger.warning("AnswerWithMemoriesTool called with empty query or memories.")
             return "Unable to answer due to missing information.", []
        
        prompt = f"""
You are an intelligent Memory Assistant. 
Please answer the following question based ONLY on the provided retrieved memories.
Question: {query}
Only output the result following the format instruction based on the retieved memories.
{self.format_instruction}

Retrieved Memories:
{memories}
"""
        response = self.llm.get_completion(prompt)
        return response, []



class MemoryRetriever:
    def __init__(self, vector_db: VectorDBBase, llm_executor: LLMAPIClientBase, config: SystemConfig):
        self.db = vector_db
        self.llm = llm_executor
        self.config = config
        self.benchmark_name = config.data_name if hasattr(config, "data_name") else "longmemeval"
        
        
        self.base_tools = [
            SearchSummaryTool(self.db),
            SearchFactsTool(self.db, config.memory.retrieval_topk),
            SearchExperiencesTool(self.db, config.memory.retrieval_topk),
            SearchPersonasTool(self.db),
            SearchTurnsTool(self.db, config.memory.retrieval_topk),
        ]
        
        self.tools = self.base_tools + [FinishTool(self.benchmark_name)]
        self.tool_map = {t.name: t for t in self.tools}
        self.collector = get_collector(config.traj_dir) 

    def get_tool_schemas(self) -> List[Dict]:
        return [t.to_schema() for t in self.tools]
    
    def construct_prompt(self, user_query, category: str = "") -> List[Dict]:
        tool_schemas = json.dumps(self.get_tool_schemas(), indent=2)
        
        
        final_result_format = get_final_result_format(self.benchmark_name, category)
        
        RETRIEVAL_SYSTEM_PROMPT = f"""
You are an intelligent Memory Assistant. Your goal is to answer the user's question by retrieving information from the database using the available tools.

[Available Tools Schema]
{tool_schemas}

[Tool Selection Guide]
- search_turns: It can retrieve the original dialogue to find the answer to the question. Turns provides the most original and pure information.
- search_facts: Best for static attributes, user profile info, habits, or current state of objects ("What is my name?", "Is the distinct constraint X active?").
- search_experiences: Best for "how-to" knowledge, reasoning patterns, or lessons learned ("How to solve error X?").
- search_summary: Use for high-level context or recap of long periods.
- search_personas: Use for retrieving the persona profile of the user.
- finish: Only when you believe that the information you have retrieved is very sufficient and you are confident to give the correct final answer, use this tool to give a very accurate answer.

[Response Format]
You must strictly follow this ReAct format:
Thought: <Analyze whether the current information is sufficient to answer the question; if not, analyze what memories need to be retrieved and which tools are needed.>
<tool_call> {{"name": "tool_name", "arguments": {{"arg_name": "value"}}}} </tool_call>

Once you receive the observation, continue the process.

[Rules]
1. The content inside <tool_call> must be a valid JSON object. And in this format: <tool_call> {{"name": "tool_name", "arguments": {{"arg_name": "value"}}}} </tool_call>.
2. You can use the search tool up to {self.config.memory.max_tool_steps} rounds. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer
3. Decompose Complex Questions: If the question involves time comparison (e.g., "time difference between X and Y") or aggregation (e.g., "total amount", "how many"), break it down. Retrieve data for X and Y separately before answering.

[Final Result Format]
When you think and call the finish tool, you must strictly follow the Final Result Format:
{final_result_format}
When you decide to give a final response, your thought process should include reasons that provide sufficient evidence to support your answer, and the thought process should also consider the requirements of the Final Result Format.
"""
        RETRIEVAL_USER_PROMPT = f"""
Please answer this question:
{user_query}
"""
        return [{"role":"system", "content":RETRIEVAL_SYSTEM_PROMPT}, {"role": "user", "content": RETRIEVAL_USER_PROMPT}]
    
    def _get_final_result_format(self, category: str = "") -> str:
        return get_final_result_format(self.benchmark_name, category)

    def _parse_xml_tool(self, text: str):
        pattern = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if not pattern:
            return None, {}
        try:
            content = pattern.group(1).strip()
            data = json.loads(content)
            return data.get("name"), data.get("arguments", {})
        except json.JSONDecodeError:
            return None, {}

    def execute_tool(self, tool_name: str, args: Dict, seen_turns: set = None) -> Tuple[str, List[Dict[str, Any]]]:
        if tool_name in self.tool_map:
            try:
                
                
                if seen_turns is not None:
                    return self.tool_map[tool_name](**args, seen_turns=seen_turns)
                else:
                    return self.tool_map[tool_name](**args)
            except Exception as e:
                logger.warning(f"Error executing {tool_name}: {e}")
                return "", []
        logger.warning(f"Tool {tool_name} not found.")
        return "", []

    def retrieve_and_answer(self, user_query: str, sample_id: str, max_steps: int = 0, category: str = "") -> Dict[str, Any]:
        if max_steps <= 0:
            max_steps = self.config.memory.max_tool_steps if self.config else 5

        
        finish_tool = FinishTool(self.benchmark_name, category)
        self.tools = self.base_tools + [finish_tool]
        self.tool_map = {t.name: t for t in self.tools}
        
        messages = self.construct_prompt(user_query, category)
        
        
        traces = []
        
        
        seen_turns = set()

        for step in range(max_steps):

            
            response_text = self.llm.get_completion(
                messages, 
                stop=["</tool_call>"], 
                json_mode=False
            )
            
            
            if "<tool_call>" in response_text and "</tool_call>" not in response_text:
                response_text += "</tool_call>"
            logger.info(f"Step {step} LLM Output: {response_text}")

            
            tool_name, tool_args = self._parse_xml_tool(response_text)
            tool_args["sample_id"] = sample_id
            if not tool_name:
                logger.warning("No tool call detected in loop.")
                messages.append({"role": "assistant", "content": "Tool Call Error"})
                messages.append({"role": "user", "content": f"""No tool call detected. Check tool call format and continue to use proper tool: <tool_call> {{"name": "tool_name", "arguments": {{"arg_name": "value"}} }} </tool_call>"""})
                continue

            
            observation, mem_metadatas = self.execute_tool(tool_name, tool_args, seen_turns=seen_turns)
            
            if not observation:
                logger.warning(f"Tool {tool_name} returned empty observation.")
                continue
            
            source_turn_ids = list(set([
                turn 
                for meta in mem_metadatas 
                if meta
                for turn_ids in meta.get("source_turn_ids", []) 
                for turn in turn_ids
            ])) 
            retrieve_op_id = str(uuid.uuid4())            
            self.collector.log_mem_step(MemTrajectoryLog(
                    sample_id=sample_id,
                    phase="retrieve",
                    session_id="",  
                    benchmark_name=self.benchmark_name,
                    op_id=retrieve_op_id,
                    input_context=messages,
                    llm_response=response_text,
                    parsed_output={"name": tool_name, "args": tool_args},
                    source_turn_ids=source_turn_ids,
                    metadata={"mem_metadatas":mem_metadatas, "user_query":user_query, "step":step}
            ))

            traces.append({
                "step": step,
                "op_id": retrieve_op_id,
                "thought": response_text.split("<tool_call>")[0].strip(),
                "tool_call": {"name": tool_name, "args": tool_args},
                "args": tool_args
            })
            
            
            if tool_name == "finish":
                return {"answer": observation, "traces": traces}
            
            
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": f"The retrieved memories are: <observation>{observation}</observation>\n Verify if you are confident enough to answer the <question>{user_query}</question> correctly and sufficiently based on the above memories; if not, continue using the tools to retrieve more information."})
        logger.warning(f"Max steps ({max_steps}) reached. Using AnswerWithMemoriesTool to answer based on collected context.")
        
        
        all_memories = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                pattern = re.search(r"<observation>(.*?)</observation>", content, re.DOTALL)
                if pattern:
                    all_memories.append(pattern.group(1).strip())
        
        full_memory_text = "\n\n".join(all_memories)
        
        
        answer_tool = AnswerWithMemoriesTool(self.llm, self.benchmark_name, category)
        answer, _ = answer_tool(user_query, full_memory_text)
        
        traces.append({
            "step": max_steps,
            "op_id": str(uuid.uuid4()),
            "thought": "Max steps reached. Invoke AnswerWithMemoriesTool.",
            "tool_call": {"name": "answer_with_memories", "args": {"query": user_query, "memories": "LEN(" + str(len(full_memory_text)) + ")"}},
            "args": {}
        })
        return {"answer": answer, "traces": traces}