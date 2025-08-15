import requests
import gradio as gr
import config
from qdrant_client import QdrantClient
from typing import Optional, Tuple, List, Dict
from urllib.parse import quote
import os

# --- Функции-обработчики для Gradio (без изменений) ---

def get_file_preview(evt: gr.SelectData):
    """
    Обработчик выбора источника из списка. Формирует iframe для отображения файла.
    """
    try:
        file_ref = evt.value[0]
        encoded_file_ref = quote(file_ref)
        file_url = f"{config.DOCS_ENDPOINT.strip('/')}/{encoded_file_ref}"
        print(f"Запрос превью для URL: {file_url}")
        iframe_html = f'<iframe src="{file_url}" width="100%" height="600px" style="border: 1px solid #ccc;"></iframe>'
        return iframe_html
    except Exception as e:
        print(f"Ошибка при обработке выбора для превью: {e}")
        return f"<p>Ошибка загрузки файла: {e}</p>"

def show_source_details_from_state(state_data: list, evt: gr.SelectData):
    """
    Обработчик выбора строки в датасете с источниками.
    (В этой версии эта функция не используется, т.к. ответ выводится в Markdown)
    """
    if not state_data or evt.index is None:
        return "*Источник не найден...*"
    
    row_index = evt.index[0]
    if row_index >= len(state_data):
        return "*Ошибка индекса. Попробуйте обновить запрос.*"

    selected_item = state_data[row_index]
    source_file = selected_item.get('source', {}).get('file', 'N/A')
    source_text = selected_item.get('source', {}).get('text', 'N/A')

    markdown_output = f"**Источник:** `{source_file}`\n\n---\n\n{source_text}"
    return markdown_output

# --- Класс Оркестратора ---

class RAGOrchestrator:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        print("✅ Клиент-оркестратор готов к работе.")

    def get_embedding(self, text: str) -> Optional[list[float]]:
        return self._make_api_request(
            config.EMBEDDING_SERVICE_ENDPOINT, 
            {"text": text}, "embedding", "сервису эмбеддингов", 60)

    def query_llm(self, question: str, context: List[dict]) -> List[dict]:
        result = self._make_api_request(
            config.OPENAI_API_ENDPOINT,
            {"question": question, "context": context},
            "answer", "LLM-сервису", 120)
        return result if isinstance(result, list) else []
    
    def _make_api_request(self, endpoint: str, payload: dict, response_key: str, service_name: str, timeout: int):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get(response_key)
        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка при обращении к {service_name}: {e}"
            print(error_msg)
            return None if response_key == "embedding" else []

    def _search_and_prepare_context(self, question_embedding: list[float]) -> Tuple[List[Dict[str, str]], list[str]]:
        search_results = self.qdrant_client.search(
            collection_name=config.COLLECTION_NAME,
            query_vector=question_embedding, limit=config.SEARCH_LIMIT, with_payload=True)
        
        if not search_results: return [], []
        
        context_chunks = [{"text": res.payload['text'], "file": res.payload['source_file']} for res in search_results]
        sources = sorted(list(set([res.payload['source_file'] for res in search_results])))
        self._log_completion(f"найдено {len(context_chunks)} фрагментов из {len(sources)} источников")
        return context_chunks, sources
    
    def _log_step(self, step_num: int, message: str): print(f"\n{step_num}. {message}")
    def _log_completion(self, message: str): print(f"   ...{message}.")

    # --- ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ - ИЗМЕНЕНА ДЛЯ ВЫВОДА В MARKDOWN ---
    def process_query_for_gradio(self, question: str):
        error_prefix = "### Ошибка\n"
        if not question:
            return "*Введите вопрос...*", gr.Dataset(samples=[]), [], "*Выберите источник...*", None

        self._log_step(1, f"Получение эмбеддинга для вопроса: '{question[:30]}...'")
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            self._log_completion("ОШИБКА")
            return error_prefix + "Не удалось получить вектор для вопроса.", gr.Dataset(samples=[]), [], "*Ошибка*", None
        self._log_completion("эмбеддинг получен")

        self._log_step(2, "Поиск релевантного контекста в Qdrant...")
        context_chunks, sources = self._search_and_prepare_context(question_embedding)
        sources_data = [[source] for source in sources]
        if not context_chunks:
            self._log_completion("контекст не найден")
            return "### Контекст не найден\nВ базе знаний не найдено релевантного контекста.", gr.Dataset(samples=sources_data), [], "*Контекст не найден*", None

        self._log_step(3, "Отправка запроса на LLM-сервис...")
        structured_answer = self.query_llm(question, context_chunks)
        if not structured_answer:
            self._log_completion("ОШИБКА")
            return error_prefix + "LLM не смог сгенерировать ответ.", gr.Dataset(samples=sources_data), [], "*Ошибка*", None
        self._log_completion("ответ от LLM получен")

        # Форматируем ответ в виде нумерованного списка для вывода в Markdown
        formatted_paragraphs = [f"{i+1}. {item['paragraph']}" for i, item in enumerate(structured_answer)]
        final_answer_string = "\n\n".join(formatted_paragraphs)
        if not final_answer_string:
            final_answer_string = "*Ответ сгенерирован, но он пустой.*"

        return final_answer_string, gr.Dataset(samples=sources_data), structured_answer, "*Кликните на источник для просмотра*", None

# --- Инициализация и запуск Gradio ---
if __name__ == "__main__":
    try:
        print("Подключение к Qdrant...")
        q_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        orchestrator = RAGOrchestrator(qdrant_client=q_client)

        print("\nЗапуск интерфейса Gradio...")
        with gr.Blocks(theme=gr.themes.Soft()) as iface:
            full_response_state = gr.State([])
            
            gr.Markdown("...") # Заголовок без изменений

            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(lines=4, label="Ваш вопрос", placeholder="Например: Каков порядок согласования командировки?")
                    submit_btn = gr.Button("Отправить", variant="primary")
                    
                    gr.Markdown("### Найденные источники (файлы)")
                    sources_box = gr.Dataset(
                        components=["text"], label="Источники", headers=["Имя файла"], samples=[["..."]])
                    
                    file_preview = gr.HTML(label="Превью документа")

                with gr.Column(scale=3):
                    # --- ИЗМЕНЕНИЕ ЗДЕСЬ: КОМПОНЕНТ ЗАМЕНЕН НА MARKDOWN ---
                    gr.Markdown("### Сгенерированный ответ")
                    answer_box = gr.Markdown(value="*Ответ появится здесь...*")

            submit_btn.click(
                fn=orchestrator.process_query_for_gradio,
                inputs=question_box,
                outputs=[answer_box, sources_box, full_response_state, file_preview] # Убрали один output, т.к. source_details_box больше не нужен
            )
            
            # Интерактивность по клику на ответ пока отключена, кликаем только на источники
            sources_box.select(
                fn=get_file_preview,
                inputs=None,
                outputs=file_preview
            )

        iface.launch(server_name="0.0.0.0", server_port=7860)

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ОРКЕСТРАТОРА: {e}")