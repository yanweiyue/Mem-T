
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from config import VectorDBConfig
from chromadb import PersistentClient, HttpClient
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from chromadb.api import ClientAPI
import json
from datetime import datetime
import uuid
from loguru import logger

class VectorDBBase(ABC):
    
    @abstractmethod
    def create_collection(self, collection_name: str, configuration = None, metadata = None, embedding_function = None, dataloader = None, get_or_create:bool= False):
        pass
    
    @abstractmethod
    def add(self, collection_name: str, ids = None, documents = None ,metadatas = None, embeddings = None):
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_embeddings = None, query_texts = None, ids = None,
               top_k: int = 10, where = None, where_document = None, include = None):
        pass
    
    @abstractmethod
    def delete(self, collection_name: str, ids = None, where = None, where_document = None):
        pass
    
    @abstractmethod
    def update(self, collection_name: str, ids, documents = None, metadatas = None, embeddings = None):
        pass

    @abstractmethod
    def get(self, collection_name: str, ids = None, where = None, where_document = None, include = None, limit: Optional[int] = None, offset: Optional[int] = None):
        pass

    @abstractmethod
    def upsert(self, collection_name: str, ids, documents = None, metadatas = None, embeddings = None):
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str):
        pass
    
class ChromaVectorDB(VectorDBBase):
    def __init__(self, config: VectorDBConfig):
        self.config = config
        if config.db_type == "persistent":
            self.client: ClientAPI = PersistentClient(path=config.path)
        elif config.db_type == "http":
            
            self.client: ClientAPI = HttpClient(host=config.host, port=config.port)
            logger.info(f"Connected to ChromaDB server at {config.host}:{config.port}")
        else:
            raise ValueError(f"Unsupported db_type: {config.db_type}. Use 'persistent' or 'http'.")
        
        
        
        logger.info("Initializing embedding function (BAAI/bge-m3)...")
        self.default_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-m3")
        logger.info("Embedding function initialized.")

    def _serialize_metadatas(self, metadatas: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        if not metadatas:
            return metadatas
        
        serialized_metadatas = []
        for meta in metadatas:
            if meta is None:
                serialized_metadatas.append(None)
                continue
            
            new_meta = {}
            for k, v in meta.items():
                if isinstance(v, (list, dict)):
                    
                    new_meta[k] = json.dumps(v, ensure_ascii=False)
                else:
                    
                    new_meta[k] = v
            serialized_metadatas.append(new_meta)
        return serialized_metadatas

    def _deserialize_metadatas(self, metadatas: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]], None]) -> Any:
        if not metadatas:
            return metadatas

        
        is_batch = isinstance(metadatas, list) and len(metadatas) > 0 and isinstance(metadatas[0], list)

        if is_batch:
            
            return [self._deserialize_metadatas(batch_item) for batch_item in metadatas]
        
        
        deserialized_metadatas = []
        for meta in metadatas:
            if meta is None:
                deserialized_metadatas.append(None)
                continue
            
            new_meta = {}
            for k, v in meta.items():
                if isinstance(v, str) and ((v.startswith('[') and v.endswith(']')) or (v.startswith('{') and v.endswith('}'))):
                    try:
                        
                        parsed_val = json.loads(v)
                        
                        
                        
                        if isinstance(parsed_val, (list, dict)):
                            new_meta[k] = parsed_val
                        else:
                            new_meta[k] = v
                    except (json.JSONDecodeError, TypeError):
                        
                        new_meta[k] = v
                else:
                    new_meta[k] = v
            deserialized_metadatas.append(new_meta)
        return deserialized_metadatas

    def create_collection(self, collection_name: str, configuration=None, metadata=None,
                           embedding_function=None, dataloader=None, get_or_create:bool=True):
        if embedding_function:
            collection = self.client.create_collection(
                name=collection_name,
                configuration=configuration,
                metadata=metadata,
                embedding_function=embedding_function,
                data_loader=dataloader,
                get_or_create=get_or_create
            )
        else:
            collection = self.client.create_collection(
                name=collection_name,
                configuration=configuration,
                metadata=metadata,
                data_loader=dataloader,
                get_or_create=get_or_create,
                embedding_function=self.default_embedding_function
            )
        return collection
    
    def add(self, collection_name: str, ids = None, documents = None, metadatas = None, embeddings = None):
        collection = self.client.get_collection(name=collection_name)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        
        clean_metadatas = self._serialize_metadatas(metadatas)

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=clean_metadatas,
            embeddings=embeddings
        )
        logger.info(f"------ Added {len(ids)} items to collection '{collection_name}' ------")
        
        
        
        logger.success(f"------ Successfully added items ------")

    def search(self, collection_name, query_embeddings=None, query_texts=None, ids=None,
               top_k = 10, where=None, where_document=None, include=None):
        collection = self.client.get_collection(name=collection_name)
        if collection.count() == 0:
            logger.warning(f"Warning: Collection '{collection_name}' is empty. Skipping query.")
            return {}
        try:
            results = collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                ids=ids,
                n_results=top_k,
                where=where,
                where_document=where_document,
                include=include
            )
            
            if results.get("metadatas"):
                results["metadatas"] = self._deserialize_metadatas(results["metadatas"])
            logger.info(f"------ Searched collection '{collection_name}' ------")
            logger.success(f"------ Successfully retrieved search results ------") 
            return results
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}' with text '{query_texts}': {e}")
            return {}

    
    def delete(self, collection_name: str, ids=None, where=None, where_document=None):
        collection = self.client.get_collection(name=collection_name)
        collection.delete(
            ids=ids,
            where=where,
            where_document=where_document
        )
        logger.info(f"------ Deleted items from collection '{collection_name}' ------")
        logger.success(f"------ Successfully deleted items ------")

    def update(self, collection_name: str, ids, documents=None, metadatas=None, embeddings=None):
        collection = self.client.get_collection(name=collection_name)
        
        
        clean_metadatas = self._serialize_metadatas(metadatas)

        collection.update(
            ids=ids,
            documents=documents,
            metadatas=clean_metadatas,
            embeddings=embeddings
        )
        logger.info(f"------ Updated items in collection '{collection_name}' ------")
        logger.success(f"------ Successfully updated items ------")

    def get(self, collection_name: str, ids=None, where=None, where_document=None, include=None, limit: Optional[int] = None, offset: Optional[int] = None):
        collection = self.client.get_collection(name=collection_name)

        if collection.count() == 0:
            print(f"Collection '{collection_name}' is empty. Adding a dummy data to initialize the index...")
            collection.add(
                documents=["A dummy note for initializing the index."],
                ids=["_init_dummy_id_"]
            )
            print("Initialization completed.")

        results = collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset
        )

        
        if results.get("metadatas"):
            results["metadatas"] = self._deserialize_metadatas(results["metadatas"])

        logger.info(f"------ Retrieved items from collection '{collection_name}' ------")
        logger.success(f"------ Successfully retrieved items ------")
        return results
    
    def upsert(self, collection_name: str, ids, documents=None, metadatas=None, embeddings=None):
        collection = self.client.get_collection(name=collection_name)

        
        clean_metadatas = self._serialize_metadatas(metadatas)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=clean_metadatas,
            embeddings=embeddings
        )
        logger.info(f"------ Upserted items in collection '{collection_name}' ------")
        logger.success(f"------ Successfully upserted items ------")

    def delete_collection(self, collection_name: str):
        collection_list = self.client.list_collections()
        collection_name_list = [collection.name for collection in collection_list]
        if collection_name in collection_name_list:
            self.client.delete_collection(name=collection_name)
            logger.info(f"------ Deleted collection '{collection_name}' ------")
            logger.success(f"------ Successfully deleted collection ------")
            return True
        else:
            logger.info(f"Collection '{collection_name}' does not exist. Skipping deletion.")
            logger.success(f"------ Successfully skipped deletion ------")
            return False

    def list_collection(self):
        collection_list = self.client.list_collections()
        collection_name_list = [collection.name for collection in collection_list]
        return collection_name_list

class VectorDBFactory:
    
    @staticmethod
    def create_db(config: VectorDBConfig) -> VectorDBBase:
        if config.backend == "chroma":
            return ChromaVectorDB(config)
        
        else:
            raise ValueError(f"VectorDB backend {config.backend} not supported")