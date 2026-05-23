"""
Agente conversacional integrado ao ChromaDB via LlamaIndex.
"""

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings
#from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
#from utils import get_env_variables

def get_chat_engine(collection_name="livros_fisica", db_path="./chroma_db"):
    #api_key, model_name = get_env_variables()
    
    #Settings.llm = OpenAI(model=model_name, api_key=api_key)
    Settings.llm = Ollama(model="qwen3.5:0.8b", base_url="http://localhost:11434")
    Settings.embed_model = OllamaEmbedding(
        model_name="embeddinggemma",
        base_url="http://localhost:11434"
    )
    
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_chat_engine(chat_mode="condense_plus_context")


