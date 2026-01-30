
from typing import Dict, Any, List
from tools_base import BaseTool
from llm_api import LLMAPIClientBase
import json
from loguru import logger



class CreateFactTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="create_fact",
            
            description=(
                "Extract 'Factual Memory' (Concrete, verifiable statements about WHAT happened).\n"
                "CRITICAL RULES:\n"
                "1. Full Entity Scan: Extract attributes and relationships between all entities mentioned (e.g., specific objects, places, third parties), not just the user.\n"
                "2. Pay special attention to time information, including relative times like 'yesterday' or 'the week before' in the text.\n"
                "Target two specific types of facts:\n"
                "1. User Factual Memory: Verifiable facts about the user's events experienced, identity, preferences, items owned, and specific constraints.\n"
                "2. Environment Factual Memory: Explicit states of the external world, object properties, document knowledge, or tool states, and other entities.\n"
            ),
            parameters={
                "fact": {
                    "type": "string", 
                    "description": "The concise, standalone declarative statement. E.g., 'The user prefers Python for backend tasks' or 'The API endpoint v2 is deprecated'."
                },
                "start_time": {
                    "type": "string", 
                    "description": "The time when the event occurred or the time when the attribute is valid. Use an empty string if it does not exist."
                },
                "end_time": {
                    "type": "string", 
                    "description": "The end time of the event or the expiration time of the attribute. Use an empty string if it does not exist."
                }
            },
            required=["fact", "start_time", "end_time"]
        )
    def __call__(self, fact: str = "", start_time: str = "", end_time: str = "", **kwargs) -> Dict:
        if not fact:
            logger.warning("CreateFactTool called with empty fact.")
            return {}
        return {"type": "fact", "document": fact, "start_time": start_time, "end_time": end_time}
        
class CreateExperienceTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="create_experience",
            description="Extract 'Experiential Memory' (Actionable lessons, patterns, or HOW-TO perform a task)\n"
                "This tool captures lessons learned, reasoning patterns, and executable skills.\n"
                "1. Strategy-based: Reusable heuristics, workflows, or insights derived from reasoning (e.g., 'To solve X, method Y is most efficient').\n"
                "2. Case-based: Key trajectories of Success or Failure that serve as examples (e.g., 'Attempting action A under condition B caused error C').\n"
                "3. Skill-based: Validated code snippets, tool usage protocols, or functions that the agent can execute.\n",
            parameters={
                "experience": {
                    "type": "string", 
                    "description": "The distilled content of the experience. It should be formulated as a rule, a cause-effect relationship, or a guideline for future actions."
                    },
                "start_time": {
                    "type": "string", 
                    "description": "The time when the event occurred or the time when the attribute is valid. Use an empty string if it does not exist."
                    },
                "end_time": {
                    "type": "string", 
                    "description": "The end time of the event or the expiration time of the attribute. Use an empty string if it does not exist."
                    }
            },
            required=["experience","start_time","end_time"]
        )
    def __call__(self, experience: str = "", start_time: str = "", end_time: str = "", **kwargs) -> Dict:
        if not experience:
            logger.warning("CreateExperienceTool called with empty experience.")
            return {}
        return {"type": "experience", "document": experience, "start_time": start_time, "end_time": end_time}

class UpdatePersonaTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="update_persona",
            description="If there is new and important information about the person, such as hobbies, participated projects or significant events, update(add or modify) the full character profile for that person.",
            parameters={
                "name": {
                    "type": "string", 
                    "description": "Name of the person. Or 'User' if the user does not have a specified name."
                    },
                "profile": {
                    "type": "string", 
                    "description": "The full, concise and updated persona text."
                    }
            },
            required=["name", "profile"]
        )
    def __call__(self, name: str = "", profile: str = "", **kwargs) -> Dict:
        if not name or not profile:
            logger.warning("UpdatePersonaTool called with empty name or profile.")
            return {}
        return {"type": "persona", "name": name, "document": profile}

class UpdateSummaryTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="update_summary",
            description="If there is new information based on the current conversation, update the runtime summary of the sessions.",
            parameters={
                "content": {
                    "type": "string",
                    "description": "The concise, complete and updated summary text."
                    }
            },
            required=["content"]
        )
    def __call__(self, content: str = "", **kwargs) -> Dict:
        if not content:
            logger.warning("UpdateSummaryTool called with empty content.")
            return {}
        return {"type": "summary", "document": content}



class MemoryFormation:
    def __init__(self, llm_executor: LLMAPIClientBase):
        self.llm_executor = llm_executor
        self.tools = [
            CreateFactTool(),
            CreateExperienceTool(),
            UpdatePersonaTool(),
            UpdateSummaryTool()
        ]
        self.tool_map = {t.name: t for t in self.tools}

    def get_tool_schemas(self) -> List[Dict]:
        return [t.to_schema() for t in self.tools]
    
    def execute_tool(self, tool_name: str, args: Dict) -> Dict:
        if tool_name in self.tool_map:
            return self.tool_map[tool_name](**args)
        return {}
    
    def construct_prompt(self, turn_text: str, prev_summary: str, prev_personas: str) -> str:
        tool_schemas = self.get_tool_schemas()
        system_prompt = f"""
You are a Memory System Analyzer. Analyze the [Current Turn] against existing memory states.
Call appropriate tools to extract information.

[Available Tools Schema]
{json.dumps(tool_schemas, indent=2)}

[Instructions]
1. Update Summary or Personas: 
- If the pervious summary and persona are empty, you MUST call `update_summary` or `update_persona` to create a new memory.
- It should contain all key information in the previous and current but remain concise.
2. Extract Facts or Experiences:
- The extracted memory should based on the current turn and not involve the previous summary and personas.
- Extract the attributes and relationships between all entities (even those not the main body of the dialogue) from the original dialogue.
3. Time Format:
Maintain the original date/time format from the input. If no format is detected, default to "15 July, 2023" style (DD Month, YYYY). 
Pay special attention to relative times like "yesterday", "last week", "last Friday" in the text: 
    + For last week/weekend/Friday, or few days ago etc, use 'the week/weekend/Friday before [certain absolute time]' to express don't calculate the exact date, just use the week/weekend/Friday before the certain absolute time, eg. 'the week/weekend/Friday before 15 July, 2023'/ few days before 15 July, 2023;
    + For last year/month/yesterday, calculate the absolute date, precise to year/month/day respectively, eg. 'July, 2023' or '19 July, 2023'.
4. **Prioritize original wording over paraphrasing.**
5. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic numerals in the answer respectively. If it is a quantity or frequency counted by yourself, default to using English.
6. Eliminate "Speech Acts". Instead, extract the information content directly. Never start a memory with "User A said/spoke/responded to B xxx.", "Speaker B asked/spoke/responded to A xxx.", or "They discussed xxx.". For example:
    * Bad: "User A said/responded to B xxx." / "User B is thinking about it."
    * Good: "User A is a vegetarian." / "The project deadline is October 15th."

[Response Format]
<tool_call> {{"name": "tool_name1", "arguments": {{"arg_name": "value"}} }} </tool_call>
<tool_call> {{"name": "tool_name2", "arguments": {{"arg_name": "value"}} }} </tool_call>
1. The content inside <tool_call> and </tool_call> must be a valid JSON object.
2. Encourage the simultaneous use of multiple tools, you can specify as many `<tool_call>  </tool_call>` statements as you like.
"""
        user_prompt = f"""
[Inputs]
Previous Summary: {prev_summary}
Previous Personas: {prev_personas}
Current Turn: {turn_text}
"""
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]