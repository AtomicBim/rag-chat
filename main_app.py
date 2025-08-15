import requests
import gradio as gr
import config
from qdrant_client import QdrantClient
from typing import Optional, Tuple, List
from urllib.parse import quote

def get_file_preview(evt: gr.SelectData):
    """
    Обработчик выбора источника из списка. Формирует iframe для отображения файла.
    """
    try:
        # У компонента Dataset evt.value - это список значений из выбранной строки.
        # Так как у нас одна колонка, берем первый элемент.
        file_ref = evt.value[0]
        
        # Кодируем имя файла для безопасной передачи в URL
        encoded_file_ref = quote(file_ref)
        
        # Формируем полный URL для доступа к файлу
        file_url = f"{config.DOCS_ENDPOINT.strip('/')}/{encoded_file_ref}"
        print(f"Запрос превью для URL: {file_url}")
        
        # Возвращаем HTML с iframe для отображения документа
        iframe_html = f'<iframe src="{file_url}" width="100%" height="600px" style="border: 1px solid #ccc;"></iframe>'
        return iframe_html
    except Exception as e:
        print(f"Ошибка при обработке выбора для превью: {e}")
        return f"<p>Ошибка загрузки файла: {e}</p>"

class RAGOrchestrator:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        print("✅ Клиент-оркестратор готов к работе.")

    def get_embedding(self, text: str) -> Optional[list[float]]:
        return self._make_api_request(
            config.EMBEDDING_SERVICE_ENDPOINT, 
            {"text": text}, 
            "embedding", 
            "сервису эмбеддингов",
            60
        )

    def query_llm(self, question: str, context: str) -> str:
        result = self._make_api_request(
            config.OPENAI_API_ENDPOINT,
            {"question": question, "context": context},
            "answer",
            "LLM-сервису",
            120
        )
        return result or "Сервер вернул пустой ответ."
    
    def _make_api_request(self, endpoint: str, payload: dict, response_key: str, service_name: str, timeout: int):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get(response_key)
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка при обращении к {service_name}: {e}"
            print(error_msg)
            return None if response_key == "embedding" else error_msg

    def process_query(self, question: str) -> Tuple[str, list, None]:
        """Полный цикл обработки вопроса от пользователя."""
        if not question:
            return "Пожалуйста, введите вопрос.", [[""]], None

        self._log_step(1, f"Получение эмбеддинга для вопроса: '{question[:30]}...'")
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Не удалось получить вектор для вопроса. Проверьте сервис эмбеддингов.", [[""]], None
        self._log_completion("эмбеддинг получен")

        self._log_step(2, "Поиск релевантного контекста в Qdrant...")
        context, sources = self._search_and_prepare_context(question_embedding)
        if not context:
            return "В базе знаний не найдено релевантного контекста.", [[""]], None

        self._log_step(3, "Отправка запроса на LLM-сервис...")
        answer = self.query_llm(question, context)
        self._log_completion("ответ от LLM получен")

        # Преобразуем список источников в формат для gr.Dataset (список списков)
        sources_data = [[source] for source in sources]
        return answer, sources_data, None
    
    # Функция _add_paperclips_to_answer полностью удалена.

    def _search_and_prepare_context(self, question_embedding: list[float]) -> Tuple[str, list[str]]:
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
        print(f"\n{step_num}. {message}")
    
    def _log_completion(self, message: str) -> None:
        print(f"   ...{message}.")

# --- Инициализация и запуск Gradio ---
if __name__ == "__main__":
    try:
        print("Подключение к Qdrant...")
        q_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

        orchestrator = RAGOrchestrator(qdrant_client=q_client)

        print("\nЗапуск интерфейса Gradio...")
        with gr.Blocks() as iface:
            gr.Markdown(
                """
                # RAG-система для ВНД Атомстройкомплекс
                Введите свой вопрос. Система найдет релевантные документы и сгенерирует ответ.
                Кликните на имя файла в списке источников для просмотра.
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(lines=3, label="Ваш вопрос к базе знаний")
                    submit_btn = gr.Button("Отправить")
                    
                    # ИЗМЕНЕНИЕ: Заменяем Textbox на Dataset для источников
                    sources_box = gr.Dataset(
                        components=["text"],
                        label="Найденные источники",
                        headers=["Имя файла"],
                        samples=[["Здесь появятся источники..."]] # Пример для отображения
                    )

                with gr.Column(scale=3):
                    # ИЗМЕНЕНИЕ: Используем Markdown для ответа, он проще и чище
                    answer_box = gr.Markdown(label="Ответ")
                    file_preview = gr.HTML(label="Превью документа")

            submit_btn.click(
                fn=orchestrator.process_query,
                inputs=question_box,
                outputs=[answer_box, sources_box, file_preview] # Направляем вывод в правильные компоненты
            )
            
            # ИЗМЕНЕНИЕ: Вешаем обработчик .select() на компонент Dataset
            sources_box.select(
                fn=get_file_preview,
                inputs=None, # Данные берутся из объекта события (evt)
                outputs=file_preview
            )

        iface.launch(server_name="0.0.0.0", server_port=7860)

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ОРКЕСТРАТОРА: {e}")