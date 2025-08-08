import requests
import gradio as gr
import config
from qdrant_client import QdrantClient

class RAGOrchestrator:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        print("✅ Клиент-оркестратор готов к работе.")

    def get_embedding(self, text: str) -> list[float] | None:
        """Получает эмбеддинг, обращаясь к сервису на GPU-машине."""
        try:
            response = requests.post(config.EMBEDDING_SERVICE_ENDPOINT, json={"text": text}, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при обращении к сервису эмбеддингов: {e}")
            return None

    def query_llm(self, question: str, context: str) -> str:
        """Обращается к LLM-сервису."""
        try:
            payload = {"question": question, "context": context}
            response = requests.post(config.OPENAI_API_ENDPOINT, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("answer", "Сервер вернул пустой ответ.")
        except requests.exceptions.RequestException as e:
            return f"Сетевая ошибка при обращении к LLM-сервису: {e}"

    def process_query(self, question: str):
        """Полный цикл обработки вопроса от пользователя."""
        if not question:
            return "Пожалуйста, введите вопрос.", ""

        print(f"\n1. Получение эмбеддинга для вопроса: '{question[:30]}...'")
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Не удалось получить вектор для вопроса. Проверьте сервис эмбеддингов.", ""
        print("   ...эмбеддинг получен.")

        print("2. Поиск релевантного контекста в Qdrant...")
        search_results = self.qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=question_embedding,
            limit=config.SEARCH_LIMIT,
            with_payload=True
        )

        if not search_results:
            return "В базе знаний не найдено релевантного контекста.", ""
        
        context = "\n---\n".join([result.payload['text'] for result in search_results])
        sources = sorted(list(set([result.payload['source_file'] for result in search_results])))
        print(f"   ...найдено {len(sources)} источников.")

        print("3. Отправка запроса на LLM-сервис...")
        answer = self.query_llm(question, context)
        print("   ...ответ от LLM получен.")

        return answer, f"Источники: {', '.join(sources)}"

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
