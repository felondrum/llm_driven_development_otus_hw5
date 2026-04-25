"""
Консольный чат-бот с интеграцией RAG системы и полным мониторингом через Langfuse.
Реализует отслеживание Traces, Spans, Generations, Events и Scores.
"""
import time
import uuid
from typing import Optional
from datetime import datetime

from langfuse import observe

import config
from monitoring import monitoring
from rag_system import RAGSystem


class ChatBot:
    """Консольный чат-бот с RAG системой и мониторингом."""

    def __init__(self):
        self.rag = RAGSystem()
        self.session_id = str(uuid.uuid4())[:8]
        self.user_id = "console_user"
        self.conversation_history = []
        self.current_trace = None

    def estimate_context_tokens(self) -> int:
        """Оценить общее количество токенов в контексте (грубая оценка)."""
        total = 0
        for msg in self.conversation_history:
            total += len(msg["query"]) // 4  # ~4 chars на токен
            total += len(msg["answer"]) // 4
        return total

    def process_query(self, user_input: str) -> dict:
        """
        Обработать запрос пользователя через RAG систему.
        
        Args:
            user_input: Входной запрос пользователя
            
        Returns:
            Словарь с результатом обработки
        """
        start_time = time.time()
        
        trace = None
        main_span = None
        rag_span = None
        llm_generation = None
        
        try:
            context_tokens = self.estimate_context_tokens()
            
            # Создать Trace для всего запроса
            trace = monitoring.create_trace(
                name="user_query_processing",
                user_id=self.user_id,
                session_id=self.session_id,
                metadata={
                    "input_length": len(user_input),
                    "conversation_turns": len(self.conversation_history),
                    "estimated_context_tokens": context_tokens,
                },
                tags=["chat", "rag"],
            )
            
            # Event: начало обработки
            monitoring.create_event(
                trace_id=trace.id,
                name="query_received",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "input_preview": user_input[:100],
                },
            )
            
            # Span: основной span для chatbot_query
            main_span = monitoring.create_span(
                trace_id=trace.id,
                name="chatbot_query",
                input_data={"query": user_input},
            )
            
            # Span: RAG обработка (вложенный в main_span)
            rag_span = monitoring.create_span(
                trace_id=trace.id,
                name="rag_processing",
                parent_observation_id=main_span.id if main_span else None,
                input_data={"query": user_input},
            )
            
            # Выполнить RAG запрос
            result = self.rag.query(user_input, trace_id=trace.id, rag_span_id=rag_span.id if rag_span else None)
            
            # Завершить RAG span с результатами
            if rag_span:
                rag_span.update(output={
                    "answer_preview": result.get("answer", "")[:100],
                    "num_docs": result.get("metrics", {}).get("num_documents_found", 0),
                    "relevance_score": result.get("metrics", {}).get("avg_relevance_score", 0.0),
                })
                rag_span.end()
            
            # Создать Generation для LLM вызова
            if trace:
                llm_generation = monitoring.create_generation(
                    trace_id=trace.id,
                    name="llm_response_generation",
                    model=config.OLLAMA_MODEL,
                    prompt=user_input,
                    completion=result["answer"],
                    usage={
                        "input": result["metrics"].get("input_tokens", 0),
                        "output": result["metrics"].get("output_tokens", 0),
                        "total": result["metrics"].get("input_tokens", 0) + result["metrics"].get("output_tokens", 0),
                    },
                    metadata={
                        "input_length": result["metrics"]["input_length"],
                        "output_length": result["metrics"]["output_length"],
                    },
                    parent_observation_id=rag_span.id if rag_span else None,
                )
                if llm_generation:
                    llm_generation.end()
            
            # Event: ответ сгенерирован
            monitoring.create_event(
                trace_id=trace.id,
                name="response_generated",
                metadata={
                    "answer_length": result["metrics"]["output_length"],
                    "documents_count": result["metrics"]["num_documents_found"],
                },
            )
            
            # Score: релевантность ответа
            if result.get("metrics", {}).get("num_documents_found", 0) > 0:
                relevance_score = min(
                    result["metrics"]["avg_relevance_score"] * 10,
                    1.0
                )
                monitoring.score(
                    trace_id=trace.id,
                    name="relevance_score",
                    value=relevance_score,
                    comment="Средняя релевантность найденных документов",
                    data_type="NUMERIC",
                )
            
            # Score: производительность
            duration_ms = result.get("metrics", {}).get("total_duration_ms", 0)
            performance_score = max(0, 1.0 - (duration_ms / 10000))  # Штраф за долгие ответы
            monitoring.score(
                trace_id=trace.id,
                name="performance_score",
                value=performance_score,
                comment=f"Время выполнения: {duration_ms:.2f}ms",
                data_type="NUMERIC",
            )
            
            # Сохранить в историю
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": user_input,
                "answer": result["answer"],
                "metrics": result["metrics"],
            })
            
            end_time = time.time()
            total_duration = (end_time - start_time) * 1000
            
            # Финальный Score: общее качество
            overall_score = (relevance_score + performance_score) / 2 if result.get("metrics", {}).get("num_documents_found", 0) > 0 else 0.5
            monitoring.score(
                trace_id=trace.id,
                name="overall_quality",
                value=overall_score,
                comment="Общее качество ответа",
                data_type="NUMERIC",
            )
            
            # Завершить main span
            if main_span:
                main_span.update(output={
                    "success": True,
                    "answer": result.get("answer", ""),
                })
                main_span.end()
            
            # Завершить trace
            monitoring.finalize_trace(trace.id, output_data={
                "success": True,
                "answer": result.get("answer", ""),
                "sources": result.get("retrieved_documents", []),
            })
            
            return {
                "success": True,
                "answer": result.get("answer", ""),
                "sources": result.get("retrieved_documents", []),
                "metrics": result.get("metrics", {}),
                "trace_id": trace.id,
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Event: ошибка
            if trace:
                monitoring.create_event(
                    trace_id=trace.id,
                    name="error_occurred",
                    metadata={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
                
                # Завершить spans с ошибкой
                if rag_span:
                    rag_span.update(output={"error": str(e)})
                    rag_span.end()
                if main_span:
                    main_span.update(output={"error": str(e)})
                    main_span.end()
                
                # Завершить trace с ошибкой
                monitoring.finalize_trace(trace.id, output_data={"error": str(e)})
            
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace.id if trace else None,
                "metrics": {
                    "total_duration_ms": (time.time() - start_time) * 1000,
                    "num_documents_found": 0,
                    "input_length": len(user_input),
                    "output_length": 0,
                }
            }

    def print_welcome(self):
        """Вывести приветственное сообщение."""
        stats = self.rag.get_document_stats()
        
        print("\n" + "=" * 60)
        print("🤖 RAG ЧАТ-БОТ ПО АРХИТЕКТУРЕ ТРАНСФОРМЕРОВ И LLM")
        print("=" * 60)
        print(f"\n📚 Загружено документов: {stats.get('total_documents', 0)}")
        print(f"🏷️ Уникальных тегов: {stats.get('unique_tags', 0)}")
        print(f"🔑 Теги: {', '.join(list(stats.get('tag_distribution', {}).keys())[:10])}")
        print("\n💬 Доступные команды:")
        print("  /help     - Показать справку")
        print("  /stats    - Статистика сессии")
        print("  /history  - История сообщений")
        print("  /quit     - Выйти из приложения")
        print("\nЗадавайте вопросы по архитектуре трансформеров и LLM!")
        print("=" * 60 + "\n")

    def print_help(self):
        """Вывести справку."""
        print("\n" + "-" * 40)
        print("СПРАВКА")
        print("-" * 40)
        print("Просто введите ваш вопрос, и я найду информацию")
        print("в базе знаний и сгенерирую ответ.")
        print("\nПримеры вопросов:")
        print("  • Как работает механизм внимания?")
        print("  • Что такое позиционное кодирование?")
        print("  • Как устроена архитектура трансформера?")
        print("  • Что такое RAG система?")
        print("  • Как работают векторные эмбеддинги?")
        print("-" * 40 + "\n")

    def print_stats(self):
        """Вывести статистику сессии."""
        print("\n" + "-" * 40)
        print("СТАТИСТИКА СЕССИИ")
        print("-" * 40)
        print(f"ID сессии: {self.session_id}")
        print(f"Всего запросов: {len(self.conversation_history)}")
        
        if self.conversation_history:
            total_input = sum(h["metrics"]["input_length"] for h in self.conversation_history)
            total_output = sum(h["metrics"]["output_length"] for h in self.conversation_history)
            avg_duration = sum(h["metrics"]["total_duration_ms"] for h in self.conversation_history) / len(self.conversation_history)
            
            print(f"Всего токенов ввода: ~{total_input}")
            print(f"Всего токенов вывода: ~{total_output}")
            print(f"Среднее время ответа: {avg_duration:.2f}ms")
            
            # Примерная стоимость
            estimated_cost = (total_input * config.TOKEN_COSTS["input"] + 
                            total_output * config.TOKEN_COSTS["output"]) / 1000
            print(f"Примерная стоимость: ${estimated_cost:.6f}")
        
        print("-" * 40 + "\n")

    def print_history(self):
        """Вывести историю сообщений."""
        if not self.conversation_history:
            print("\nИстория пуста.\n")
            return
        
        print("\n" + "=" * 60)
        print("ИСТОРИЯ СООБЩЕНИЙ")
        print("=" * 60)
        
        for i, entry in enumerate(self.conversation_history[-5:], 1):  # Последние 5
            print(f"\n[{i}] {entry['timestamp'][:19]}")
            print(f"   Q: {entry['query'][:80]}...")
            print(f"   A: {entry['answer'][:120]}...")
            print(f"   ⏱️ {entry['metrics']['total_duration_ms']:.2f}ms | "
                  f"📄 {entry['metrics']['num_documents_found']} док.")
        
        print("=" * 60 + "\n")

    def run(self):
        """Запустить консольный чат-бот."""
        self.print_welcome()

        try:
            while True:
                try:
                    user_input = input("\n👤 Вы: ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                # Обработка команд
                if user_input.lower() in ["/quit", "/exit", "/q"]:
                    print("\n👋 До свидания!")
                    break
                
                elif user_input.lower() in ["/help", "/h", "?"]:
                    self.print_help()
                    continue
                
                elif user_input.lower() == "/stats":
                    self.print_stats()
                    continue
                
                elif user_input.lower() == "/history":
                    self.print_history()
                    continue

                # Обработка запроса
                print("\n🤖 Бот: Обрабатываю запрос...")
                result = self.process_query(user_input)

                if result and result.get("success"):
                    print(f"\n🤖 Бот: {result['answer']}")
                    
                    if result["sources"]:
                        print("\n📚 Источники:")
                        for source in result["sources"]:
                            print(f"  • {source['title']} (релевантность: {source['score']:.3f})")
                    
                    print(f"\n⏱️ Время: {result['metrics']['total_duration_ms']:.2f}ms | "
                          f"📄 Найдено документов: {result['metrics']['num_documents_found']}")
                else:
                    error_msg = result.get('error', 'Неизвестная ошибка') if result else 'Неизвестная ошибка'
                    print(f"\n❌ Ошибка: {error_msg}")

        except KeyboardInterrupt:
            print("\n\n👋 Прервано пользователем. До свидания!")
        
        finally:
            # Очистка ресурсов
            monitoring.flush()
            monitoring.shutdown()


def main():
    """Точка входа в приложение."""
    bot = ChatBot()
    bot.run()


if __name__ == "__main__":
    main()
