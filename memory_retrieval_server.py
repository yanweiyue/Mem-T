import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Set
import os
import sys
from contextlib import asynccontextmanager


sys.path.append(os.getcwd())

from config import SystemConfig
from vector_db import VectorDBFactory
from memory_retrieval import MemoryRetriever
from llm_api import LLMAPIClientBase


class MockLLM(LLMAPIClientBase):
    def get_completion(self, *args, **kwargs):
        return "Mock response"


retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Initializing Memory Retrieval Server...")
    config = SystemConfig()
    
    
    try:
        vector_db = VectorDBFactory.create_db(config.vector_db)
        print("Vector DB initialized.")
    except Exception as e:
        print(f"Failed to init Vector DB: {e}")
        raise e

    
    llm = MockLLM()
    
    retriever = MemoryRetriever(vector_db=vector_db, llm_executor=llm, config=config)
    print("MemoryRetriever initialized.")
    yield    
    print("Shutting down Memory Retrieval Server...")


app = FastAPI(lifespan=lifespan)

class RetrieveRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    sample_id: str
    seen_turns: List[str] = [] 

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    global retriever
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    
    try:
        seen_turns_set = set(request.seen_turns)
        request.arguments["sample_id"] = request.sample_id
        
        
        observation, mem_metadatas = retriever.execute_tool(
            request.tool_name, 
            request.arguments, 
            seen_turns=seen_turns_set
        )
        
        return {
            "result": observation,
            "mem_metadatas": mem_metadatas,
            "updated_seen_turns": list(seen_turns_set)
        }
    except Exception as e:
        print(f"Error executing tool {request.tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)