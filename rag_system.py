"""
RAG система для работы с документами по архитектуре трансформеров и LLM.
Использует локальную векторную базу данных и Ollama для эмбеддингов и генерации.
"""
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter

import config
from monitoring import monitoring, track_operation


@dataclass
class RetrievalResult:
    """Результат поиска документа."""
    doc_id: str
    title: str
    content: str
    score: float
    tags: List[str]


class RAGSystem:
    """RAG система для поиска и генерации ответов на основе документов."""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine: Optional[RetrieverQueryEngine] = None
        self._setup_models()
        self._load_documents()

    def _setup_models(self):
        """Настроить модели LLM и эмбеддингов."""
        Settings.embed_model = OllamaEmbedding(
            model_name=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=120.0,
        )
        Settings.text_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

    def _load_documents(self):
        """Загрузить документы из JSON файла."""
        try:
            with open(config.DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
            self._build_index()
        except FileNotFoundError:
            print(f"Файл с данными не найден: {config.DATA_PATH}")
            self.documents = []
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}")
            self.documents = []

    def _build_index(self):
        """Построить векторный индекс из документов."""
        if not self.documents:
            return

        # Создать документы для llama-index
        llama_docs = []
        for doc in self.documents:
            content = f"Заголовок: {doc['title']}\nСодержание: {doc['content']}"
            metadata = {
                "doc_id": doc["id"],
                "title": doc["title"],
                "tags": doc.get("tags", []),
            }
            llama_docs.append(Document(text=content, metadata=metadata))

        # Построить индекс
        self.index = VectorStoreIndex.from_documents(
            llama_docs,
            show_progress=True,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=config.TOP_K_RETRIEVAL,
        )

    @track_operation
    def search(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию TOP_K_RETRIEVAL из конфига)
            
        Returns:
            Список найденных документов с метаданными
        """
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        if not self.index:
            return []

        start_time = time.time()
        
        # Выполнить поиск
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        nodes = retriever.retrieve(query)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Обработать результаты
        results = []
        for node in nodes:
            metadata = node.metadata
            results.append(RetrievalResult(
                doc_id=metadata.get("doc_id", "unknown"),
                title=metadata.get("title", "Unknown"),
                content=node.text,
                score=node.score if hasattr(node, "score") else 0.0,
                tags=metadata.get("tags", []),
            ))

        # Логирование в Langfuse
        monitoring.create_event(
            trace_id="rag-search",
            name="documents_retrieved",
            metadata={
                "query": query,
                "num_results": len(results),
                "duration_ms": duration_ms,
                "top_k": top_k,
            },
        )

        return results

    @track_operation
    def generate_answer(
        self,
        query: str,
        context_docs: List[RetrievalResult],
    ) -> str:
        """
        Сгенерировать ответ на основе контекста.
        
        Args:
            query: Исходный запрос пользователя
            context_docs: Найденные релевантные документы
            
        Returns:
            Сгенерированный ответ
        """
        if not context_docs:
            return "К сожалению, я не нашёл релевантной информации по вашему запросу."

        # Построить контекст из найденных документов
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"[Источник {i}: {doc.title}]\n{doc.content}"
            )
        context = "\n\n".join(context_parts)

        # Системный промпт
        system_prompt = f"""Ты — экспертный помощник по архитектуре трансформеров и большим языковым моделям (LLM).
Отвечай на вопросы точно и информативно, используя предоставленный контекст.
Если информация в контексте недостаточна, честно скажи об этом.

Контекст:
{context}

Вопрос пользователя: {query}

Ответ:"""

        start_time = time.time()
        
        # Вызов LLM через llama-index
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=query),
        ]
        
        response = Settings.llm.chat(messages)
        answer = response.message.content or ""
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Подсчитать токены (примерно)
        input_tokens = len(system_prompt.split()) + len(query.split())
        output_tokens = len(answer.split())
        
        # Логирование генерации в Langfuse
        generation = monitoring.create_generation(
            trace_id="rag-generation",
            name="llm_answer_generation",
            model=config.OLLAMA_MODEL,
            prompt=system_prompt,
            completion=answer,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            metadata={
                "num_context_docs": len(context_docs),
                "context_doc_ids": [d.doc_id for d in context_docs],
                "duration_ms": duration_ms,
            },
        )

        return answer

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Полный цикл RAG: поиск + генерация ответа.
        
        Args:
            user_query: Запрос пользователя
            
        Returns:
            Словарь с ответом и метаданными
        """
        start_time = time.time()

        # Поиск документов
        retrieved_docs = self.search(user_query)

        # Генерация ответа
        answer = self.generate_answer(user_query, retrieved_docs)

        end_time = time.time()
        total_duration_ms = (end_time - start_time) * 1000

        # Рассчитать релевантность (средний скор найденных документов)
        avg_relevance = sum(d.score for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0

        return {
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": d.doc_id,
                    "title": d.title,
                    "score": d.score,
                    "tags": d.tags,
                }
                for d in retrieved_docs
            ],
            "metrics": {
                "num_documents_found": len(retrieved_docs),
                "avg_relevance_score": avg_relevance,
                "total_duration_ms": total_duration_ms,
                "input_length": len(user_query),
                "output_length": len(answer),
            },
        }

    def get_document_stats(self) -> Dict[str, Any]:
        """Получить статистику по загруженным документам."""
        if not self.documents:
            return {"total_documents": 0}

        all_tags = []
        for doc in self.documents:
            all_tags.extend(doc.get("tags", []))

        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_documents": len(self.documents),
            "unique_tags": len(set(all_tags)),
            "tag_distribution": tag_counts,
        }
