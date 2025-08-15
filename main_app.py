import requests
import gradio as gr
import config
from qdrant_client import QdrantClient
from typing import Optional, Tuple

class RAGOrchestrator:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        print("✅ Клиент-оркестратор готов к работе.")

    def get_embedding(self, text: str) -> Optional[list[float]]:
        """Получает эмбеддинг, обращаясь к сервису на GPU-машине."""
        return self._make_api_request(
            config.EMBEDDING_SERVICE_ENDPOINT, 
            {"text": text}, 
            "embedding", 
            "сервису эмбеддингов",
            60
        )

    def query_llm(self, question: str, context: str) -> str:
        """Обращается к LLM-сервису."""
        result = self._make_api_request(
            config.OPENAI_API_ENDPOINT,
            {"question": question, "context": context},
            "answer",
            "LLM-сервису",
            120
        )
        return result or "Сервер вернул пустой ответ."
    
    def _make_api_request(self, endpoint: str, payload: dict, response_key: str, service_name: str, timeout: int):
        """Общий метод для выполнения API-запросов."""
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get(response_key)
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка при обращении к {service_name}: {e}"
            print(error_msg)
            return None if response_key == "embedding" else error_msg

    def process_query(self, question: str) -> Tuple[str, str]:
        """Полный цикл обработки вопроса от пользователя."""
        if not question:
            return "Пожалуйста, введите вопрос.", ""

        self._log_step(1, f"Получение эмбеддинга для вопроса: '{question[:30]}...'")
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Не удалось получить вектор для вопроса. Проверьте сервис эмбеддингов.", ""
        self._log_completion("эмбеддинг получен")

        self._log_step(2, "Поиск релевантного контекста в Qdrant...")
        context, sources = self._search_and_prepare_context(question_embedding)
        if not context:
            return "В базе знаний не найдено релевантного контекста.", ""

        self._log_step(3, "Отправка запроса на LLM-сервис...")
        answer = self.query_llm(question, context)
        self._log_completion("ответ от LLM получен")

        return answer, f"Источники: {', '.join(sources)}"
    
    def _search_and_prepare_context(self, question_embedding: list[float]) -> Tuple[str, list[str]]:
        """Поиск контекста в Qdrant и подготовка источников."""
        search_results = self.qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=question_embedding,
            limit=config.SEARCH_LIMIT,
            with_payload=True
        )
        
        if not search_results:
            return "", []
        
        context = "\n---\n".join([result.payload['text'] for result in search_results])
        sources = sorted(list(set([result.payload['source_file'] for result in search_results])))
        self._log_completion(f"найдено {len(sources)} источников")
        return context, sources
    
    def _log_step(self, step_num: int, message: str) -> None:
        """Логирование шага обработки."""
        print(f"\n{step_num}. {message}")
    
    def _log_completion(self, message: str) -> None:
        """Логирование завершения шага."""
        print(f"   ...{message}.")

# --- Инициализация и запуск Gradio ---
if __name__ == "__main__":
    try:
        print("Подключение к Qdrant...")
        q_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        
        orchestrator = RAGOrchestrator(qdrant_client=q_client)

        print("\nЗапуск интерфейса Gradio...")
        iface = gr.Interface(
            fn=orchestrator.process_query,
            inputs=gr.Textbox(lines=3, label="Ваш вопрос к базе знаний"),
            outputs=[
                gr.Textbox(label="Ответ"),
                gr.Textbox(label="Найденные источники")
            ],
            title="RAG-система для ВНД Атомстройкомплекс ",
            description="Введите свой вопрос. Система наидет релевантные документы и сгенерирует ответ."
        )
        
        # Запускаем Gradio на порту 80, чтобы был доступен по IP машины
        iface.launch(server_name="0.0.0.0")

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ОРКЕСТРАТОРА: {e}")
