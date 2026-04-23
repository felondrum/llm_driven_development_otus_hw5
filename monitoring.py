"""
Модуль для инициализации Langfuse клиента и инструментов мониторинга.
Реализует отслеживание Traces, Spans, Generations, Events и Scores.
"""
from langfuse import Langfuse, observe
import time
from typing import Optional, Dict, Any, List
import config


class MonitoringService:
    """Сервис для мониторинга LLM приложений через Langfuse."""

    def __init__(self):
        self.client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """Создать новый Trace для отслеживания полного пути запроса."""
        return self.client.trace(
            name=name,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )

    def create_span(
        self,
        trace_id: str,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
    ):
        """Создать Span для отдельной операции."""
        span = self.client.span(
            trace_id=trace_id,
            name=name,
            metadata=metadata or {},
            input=input_data,
            output=output_data,
        )
        if start_time:
            span.update(start_time=start_time)
        if end_time:
            span.end(end_time=end_time)
        return span

    def create_generation(
        self,
        trace_id: str,
        name: str,
        model: str,
        prompt: Any,
        completion: Any,
        usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        """Создать Generation для LLM вызова."""
        generation = self.client.generation(
            trace_id=trace_id,
            name=name,
            model=model,
            input=prompt,
            output=completion,
            usage=usage,
            metadata=metadata or {},
        )
        if start_time:
            generation.update(start_time=start_time)
        if end_time:
            generation.end(end_time=end_time)
        return generation

    def create_event(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Создать Event для точечного события."""
        return self.client.event(
            trace_id=trace_id,
            name=name,
            metadata=metadata or {},
        )

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
    ):
        """Добавить Score (метрику качества/производительности)."""
        return self.client.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
        )

    def flush(self):
        """Принудительно отправить все данные в Langfuse."""
        self.client.flush()

    def shutdown(self):
        """Корректно завершить работу клиента."""
        self.client.shutdown()


# Глобальный экземпляр сервиса
monitoring = MonitoringService()


def track_llm_call(func):
    """Декоратор для автоматического отслеживания вызовов LLM."""
    @observe(name=func.__name__, as_type="generation")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def track_operation(func):
    """Декоратор для автоматического отслеживания операций."""
    @observe(name=func.__name__, as_type="span")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
