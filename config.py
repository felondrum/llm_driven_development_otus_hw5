"""
Конфигурация приложения для мониторинга LLM через Langfuse.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Langfuse конфигурация
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-1234567890")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-1234567890")

# Ollama конфигурация
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# RAG конфигурация
DATA_PATH = os.getenv("DATA_PATH", "data/data.json")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))

# Стоимость токенов (примерная, $ за 1M токенов)
TOKEN_COSTS = {
    "input": 0.0001,  # $ per 1K input tokens
    "output": 0.0003,  # $ per 1K output tokens
}
