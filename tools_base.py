
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

class BaseTool(ABC):
    def __init__(self, 
                 name: str, 
                 description: str, 
                 parameters: Dict[str, Any],
                 required: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        
        self.required = required if required is not None else list(parameters.keys())

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        pass