"""
Модуль для инициализации Langfuse клиента и инструментов мониторинга.
Реализует отслеживание Traces, Spans, Generations, Events и Scores.
"""
from langfuse import Langfuse, observe
import time
from typing import Optional, Dict, Any, List, Union
import config


# Типы для возвращаемых значений
TraceClient = Any  # В langfuse v4 это обёртка над observation
SpanClient = Any
GenerationClient = Any
EventClient = Any


class MonitoringService:
    """Сервис для мониторинга LLM приложений через Langfuse."""

    def __init__(self):
        self.client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        # Хранилище активных trace для корректной работы иерархии
        self._active_traces: Dict[str, TraceClient] = {}

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> TraceClient:
        """Создать новый Trace для отслеживания полного пути запроса."""
        full_metadata = (metadata or {}).copy()
        if user_id:
            full_metadata["user_id"] = user_id
        if tags:
            full_metadata["tags"] = tags
        
        # В Langfuse v4 trace создаётся через start_observation с as_type='trace'
        trace = self.client.start_observation(
            name=name,
            as_type="trace",
            metadata=full_metadata,
        )
        
        # Сохраняем trace для последующего использования
        self._active_traces[trace.id] = trace
        
        return trace

    def get_trace(self, trace_id: str) -> Optional[TraceClient]:
        """Получить активный trace по ID."""
        return self._active_traces.get(trace_id)

    def create_span(
        self,
        trace_id: str,
        name: str,
        parent_observation_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input_data: Optional[Any] = None,
        output_data: Optional[Any] = None,
    ) -> Optional[SpanClient]:
        """Создать Span для отдельной операции."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            print(f"[WARNING] Trace {trace_id} not found for span '{name}'")
            return None
        
        # В Langfuse v4 используем start_observation с trace_context
        span = self.client.start_observation(
            name=name,
            as_type="span",
            metadata=metadata or {},
            input=input_data,
            trace_context={"trace_id": trace_id},
        )
        
        if start_time:
            span.update(start_time=start_time)
        if end_time:
            span.end()
        elif input_data is not None and output_data is not None:
            # Если есть входные и выходные данные, завершаем span
            span.end()
        
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
        parent_observation_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Optional[GenerationClient]:
        """Создать Generation для LLM вызова."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            print(f"[WARNING] Trace {trace_id} not found for generation '{name}'")
            return None
        
        # Форматируем usage для Langfuse
        usage_details = {}
        if usage:
            if "input" in usage:
                usage_details["promptTokens"] = usage["input"]
            if "output" in usage:
                usage_details["completionTokens"] = usage["output"]
            if "total" in usage:
                usage_details["totalTokens"] = usage["total"]
        
        generation = self.client.start_observation(
            name=name,
            as_type="generation",
            model=model,
            input=prompt,
            output=completion,
            usage=usage_details if usage_details else None,
            metadata=metadata or {},
            trace_context={"trace_id": trace_id},
        )
        
        if start_time:
            generation.update(start_time=start_time)
        if end_time:
            generation.end()
        
        return generation

    def create_event(
        self,
        trace_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        parent_observation_id: Optional[str] = None,
    ) -> Optional[EventClient]:
        """Создать Event для точечного события."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            print(f"[WARNING] Trace {trace_id} not found for event '{name}'")
            return None
        
        # В Langfuse v4 events создаются через start_observation с as_type='event'
        event = self.client.start_observation(
            name=name,
            as_type="event",
            metadata=metadata or {},
            trace_context={"trace_id": trace_id},
        )
        
        return event

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
        observation_id: Optional[str] = None,
    ):
        """Добавить Score (метрику качества/производительности)."""
        trace = self._active_traces.get(trace_id)
        if not trace:
            print(f"[WARNING] Trace {trace_id} not found for score '{name}'")
            return
        
        # В Langfuse v4 используем create_score
        self.client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type,
        )

    def finalize_trace(self, trace_id: str, output_data: Optional[Any] = None):
        """Завершить trace и удалить из активного хранилища."""
        trace = self._active_traces.get(trace_id)
        if trace:
            trace.end()
            # Удаляем из активных
            del self._active_traces[trace_id]

    def flush(self):
        """Принудительно отправить все данные в Langfuse."""
        self.client.flush()

    def shutdown(self):
        """Корректно завершить работу клиента."""
        # Завершаем все активные traces
        for trace_id in list(self._active_traces.keys()):
            self.finalize_trace(trace_id)
        self.client.shutdown()


# Глобальный экземпляр сервиса
monitoring = MonitoringService()


def track_llm_call(func):
    """Декоратор для автоматического отслеживания вызовов LLM."""
    @observe(name=func.__name__, as_type="generation")
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# def track_operation(func):
#     """Декоратор для автоматического отслеживания операций."""
#     @observe(name=func.__name__, as_type="span")
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs)
#     return wrapper
