
import uuid
import json
from typing import Dict, Any, List
from loguru import logger
from vector_db import VectorDBBase
from llm_api import LLMAPIClientBase
from tools_base import BaseTool
import time
from utils import format_memory_content, parse_memory_content



class AddItemTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, collection_name: str):
        super().__init__(
            name="add_item",
            description="Add a new memory item.",
            parameters={
                "document": {"type": "string", "description": "The content."},
                "turn_time": {"type": "string", "description": "The time of the turn that generated this item."},
                "start_time": {"type": "string", "description": "Start time."},
                "end_time": {"type": "string", "description": "End time."},
            },
            required=["document"]
        )
        self.db = vector_db
        self.collection_name = collection_name

    def __call__(self, document: str, turn_time: str = "", start_time: str = "", end_time: str = "", original_text: str = "", original_turns: List[str] = None, **kwargs):
        if not document:
            logger.warning("AddItemTool called with empty document.")
            return {"action": "ADD", "id": "", "status": "failed", "reason": "document is empty"}
        item_id = str(uuid.uuid4())
        metadatas = [{"turn_time": turn_time, "start_time": start_time, "end_time": end_time}]
        
        
        if original_turns:
             metadatas[0]["original_turns"] = original_turns
             
             metadatas[0]["memory_content"] = document

        
        if "personas" in self.collection_name and kwargs.get("name"):
            item_id = kwargs.get("name") 
        
        
        if kwargs.get("source_turn_ids"):
            source_turn_ids = kwargs.get("source_turn_ids")
            metadatas[0]["source_turn_ids"] = [source_turn_ids]
        else:
            metadatas[0]["source_turn_ids"] = []
        
        if kwargs.get("op_ids"):
            op_ids = kwargs.get("op_ids")
            metadatas[0]["op_ids"] = [op_ids]
        else:
            metadatas[0]["op_ids"] = []  
        metadatas[0]["id"] = item_id
        metadatas[0]["col_name"] = self.collection_name
        
        
        
        source_text = ""
        if original_turns:
            source_text = "\n".join(original_turns)
        elif original_text:
            source_text = original_text
            
        full_document = format_memory_content(document, source_text)
        
        self.db.add(self.collection_name, ids=[item_id], documents=[full_document], metadatas=metadatas)
        return {"action": "ADD", "id": item_id, "status": "success"}

class UpdateItemTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, collection_name: str):
        super().__init__(
            name="update_item",
            description="Update an existing memory item.",
            parameters={
                "id": {"type": "string", "description": "The ID of the item to update."},
                "document": {"type": "string", "description": "Enrich the content with more details and update the statistical data or factual frequencies mentioned. Must save the original time information of previously items in the document."},
                "turn_time": {"type": "string", "description": "The time of the turn that generated this update."},
                "start_time": {"type": "string", "description": "New start time."},
                "end_time": {"type": "string", "description": "New end time."},
            },
            required=["id", "document"]
        )
        self.db = vector_db
        self.collection_name = collection_name

    def __call__(self, id: str = "", document: str = "", turn_time: str = "", start_time: str = "", end_time: str = "", original_text: str = "", original_turns: List[str] = None, **kwargs):
        if not id or not document:
            logger.warning("UpdateItemTool called with empty id or document.")
            return {"action": "UPDATE", "id": id, "status": "failed", "reason": "id or document is empty"}
        metadatas = [{"turn_time": turn_time, "start_time": start_time, "end_time": end_time}]
        
        
        existing_items = self.db.get(self.collection_name, ids=[id], include=["metadatas", "documents"])
        existing_source = ""
        existing_original_turns = []
        
        if existing_items and "documents" in existing_items and len(existing_items["documents"]) > 0:
             existing_doc = existing_items["documents"][0]
             _, existing_source = parse_memory_content(existing_doc)
        
        if existing_items and "metadatas" in existing_items and len(existing_items["metadatas"]) > 0:
            existing_meta = existing_items["metadatas"][0]
            if existing_meta and "original_turns" in existing_meta:
                existing_original_turns = existing_meta["original_turns"]
                
                if isinstance(existing_original_turns, str):
                    try:
                        existing_original_turns = json.loads(existing_original_turns)
                    except:
                        existing_original_turns = [existing_original_turns]
                elif not isinstance(existing_original_turns, list):
                     existing_original_turns = [str(existing_original_turns)]

        
        current_turns = []
        if original_turns:
            current_turns = original_turns
        elif original_text:
            current_turns = [original_text]
            
        new_original_turns = list(set(existing_original_turns + current_turns))
        metadatas[0]["original_turns"] = new_original_turns
        metadatas[0]["memory_content"] = document

        if kwargs.get("source_turn_ids"):
            source_turn_ids = kwargs.get("source_turn_ids") 
            if existing_items and "metadatas" in existing_items and len(existing_items["metadatas"]) > 0 and "source_turn_ids" in existing_items["metadatas"][0]:
                existing_metadata = existing_items["metadatas"][0]
                existing_source_turn_ids = existing_metadata["source_turn_ids"] 
                
                if isinstance(existing_source_turn_ids, list) and not isinstance(existing_source_turn_ids[0], list):
                    existing_source_turn_ids = [existing_source_turn_ids]
                existing_source_turn_ids.append(source_turn_ids)
                source_turn_ids = existing_source_turn_ids
            else:
                source_turn_ids = [source_turn_ids]
            metadatas[0]["source_turn_ids"] = source_turn_ids
        if kwargs.get("op_ids"):
            op_ids = kwargs.get("op_ids") 
            if existing_items and "metadatas" in existing_items and len(existing_items["metadatas"]) > 0 and "op_ids" in existing_items["metadatas"][0]:
                existing_metadata = existing_items["metadatas"][0]
                existing_op_ids = existing_metadata["op_ids"] 
                
                if isinstance(existing_op_ids, list) and not isinstance(existing_op_ids[0], dict):
                    existing_op_ids = [existing_op_ids]
                existing_op_ids.append(op_ids)
                op_ids = existing_op_ids
            else:
                op_ids = [op_ids]
            metadatas[0]["op_ids"] = op_ids
        metadatas[0]["id"] = id
        metadatas[0]["col_name"] = self.collection_name
        
        
        
        
        new_source = "\n".join(new_original_turns)
        
        full_document = format_memory_content(document, new_source)
        
        self.db.upsert(self.collection_name, ids=[id], documents=[full_document], metadatas=metadatas)
        return {"action": "UPDATE", "id": id, "status": "success"}

class DeleteItemTool(BaseTool):
    def __init__(self, vector_db: VectorDBBase, collection_name: str):
        super().__init__(
            name="delete_item",
            description="Delete an existing memory item. Use when an item is explicitly negated or wrong.",
            parameters={
                "id": {"type": "string", "description": "The ID to delete."}
            },
            required=["id"]
        )
        self.db = vector_db
        self.collection_name = collection_name

    def __call__(self, id: str, **kwargs):
        if not id:
            logger.warning("DeleteItemTool called with empty id.")
            return {"action": "DELETE", "id": id, "status": "failed", "reason": "id is empty"}
        self.db.delete(self.collection_name, ids=[id])
        return {"action": "DELETE", "id": id, "status": "success"}

class IgnoreItemTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="ignore_item",
            description="Do nothing. If the item is completely redundant in both *semantic meaning* and *time range*.",
            parameters={
                 "reason": {"type": "string", "description": "Reason for ignoring."}
            },
            required=["reason"]
        )

    def __call__(self, reason: str = "", **kwargs):
        return {"action": "IGNORE", "reason": reason, "id":"", "status":"success"}


class MemoryUpdate:
    def __init__(self, llm_executor: LLMAPIClientBase, vector_db: VectorDBBase):
        self.llm_executor = llm_executor
        self.db = vector_db
        self.tools_cache = {} 

    def _get_tools(self, collection_name: str) -> List[BaseTool]:
        if collection_name not in self.tools_cache:
            self.tools_cache[collection_name] = [
                AddItemTool(self.db, collection_name),
                UpdateItemTool(self.db, collection_name),
                DeleteItemTool(self.db, collection_name),
                IgnoreItemTool()
            ]
        return self.tools_cache[collection_name]

    def get_tool_schemas(self, collection_name: str) -> List[Dict]:
        return [t.to_schema() for t in self._get_tools(collection_name)]
    
    def execute_tool(self, collection_name: str, tool_name: str, args: Dict) -> Dict:
        tools = self._get_tools(collection_name)
        for t in tools:
            if t.name == tool_name:
                result = t(**args)
                if result and "status" in result and result["status"] == "success":
                    return result
                else:
                    logger.warning(f"Tool {tool_name} returned empty result.")
                    return {"status": "failed", "reason": "Tool returned empty result"}
        return {"status": "failed", "reason": "Tool not found"}

    def construct_prompt(self, potential_item: Dict, related_items: List[Dict], tool_schemas: List[Dict]) -> str:
        system_prompt = f"""
You are an Insights Database Administrator. Decide how to process a 'Potential Memory Item' based on 'Related Items' and the available tools. 

[Available Tools Schema]
{json.dumps(tool_schemas, indent=2)}

[Decision Logic]
Compare the [New Information] against [Existing Memories] and apply the following logic strictly:
The memory should be atomic. Each memory represents only a single piece of information. It needs to be brief and contain only a single message.

**Step 1: Check for Contradiction**
- Does the new info explicitly say the old info is completely wrong? 
- -> If YES: **DELETE**.

**Step 2: Check for Identity**
- Is the new info exactly the same with existing memories (content + time)? 
- -> If YES: **IGNORE**.

**Step 3: Check for Atomic Updates**
- Only new information explicitly overwrites a single mutable field (count, quantity, status, boolean, time) of the same entity, where the new value is **unambiguous and fully replaces the old one**. 
- e.g. "I have 4 models" + "Bought one more" → "I have 5 models"
- -> If YES: **UPDATE**.
- Only use **UPDATE** when the change is a simple refinement, correction, or state update that makes the old version obsolete.
  For two separate pieces of information like this, never use `update`. Instead, split them into two separate memories and use ADD tool to add them separately. For example: "1. Jon is passionate about dancing and has been involved in dancing since childhood." "2. He is planning to start a dance studio." 
When you update, the original memory entries' timestamps should be inserted into the document to prevent them from being overwritten during the update.

**Step 4: Default Fallback - Prefer ADD over UPDATE**
- If the event happened at a different time (recurrence), involves a different person, or introduces a new topic:
- -> **ADD**.
- **IMPORTANT**: If the new information adds significant context or details (e.g. a new event, a specific instance of a habit, or a new reason), prefer **ADD** to keep the history clear and avoid merging distinct events into one long text.

**CRITICAL RULES:**
- Do NOT merge separate occurrences of similar habits (e.g., "Running on Monday" and "Running on Friday" must be two ADDs, not one UPDATE).
- When in doubt, **ADD**.

[Response Format]
<tool_call> {{"name": "tool_name", "arguments": {{"arg_name": "value"}} }} </tool_call>  
"""
        user_prompt = f"""
[New memory to be added]
{json.dumps(potential_item, ensure_ascii=False, indent=2)}

[Existing memories used to decide the action]
{json.dumps(related_items, ensure_ascii=False, indent=2)}
"""
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
