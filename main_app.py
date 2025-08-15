import requests
import gradio as gr
import config
from qdrant_client import QdrantClient
from typing import Optional, Tuple, List, Dict, Iterator
from urllib.parse import quote
import os

# --- Функции-обработчики для Gradio ---

def get_file_preview(evt: gr.SelectData):
    try:
        file_ref = evt.value[0]
        encoded_file_ref = quote(file_ref)
        file_url = f"{config.DOCS_ENDPOINT.strip('/')}/{encoded_file_ref}"
        return f'<iframe src="{file_url}" width="100%" height="600px" style="border: 1px solid #ccc;"></iframe>'
    except Exception as e:
        return f"<p>Ошибка загрузки файла: {e}</p>"

def show_source_details_from_state(state_data: list, evt: gr.SelectData):
    if not state_data or evt.index is None:
        return "*Источник не найден...*"
    
    row_index = evt.index[0]
    if row_index >= len(state_data):
        return "*Ошибка индекса. Попробуйте обновить запрос.*"

    item = state_data[row_index]
    source_file = item.get('source', {}).get('file', 'N/A')
    source_text = item.get('source', {}).get('text', 'N/A')
    return f"**Источник:** `{source_file}`\n\n---\n\n{source_text}"

# --- Класс Оркестратора ---

class RAGOrchestrator:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        print("✅ Клиент-оркестратор готов к работе.")

    # ... (методы get_embedding, query_llm, _make_api_request, _search_and_prepare_context, _log_step, _log_completion без изменений) ...
    def get_embedding(self, text: str) -> Optional[list[float]]:
        return self._make_api_request(config.EMBEDDING_SERVICE_ENDPOINT, {"text": text}, "embedding", "сервису эмбеддингов", 60)
    def query_llm(self, question: str, context: List[dict]) -> List[dict]:
        result = self._make_api_request(config.OPENAI_API_ENDPOINT, {"question": question, "context": context}, "answer", "LLM-сервису", 120)
        return result if isinstance(result, list) else []
    def _make_api_request(self, endpoint: str, payload: dict, response_key: str, service_name: str, timeout: int):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get(response_key)
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при обращении к {service_name}: {e}")
            return None if response_key == "embedding" else []
    def _search_and_prepare_context(self, q_embedding: list[float]) -> Tuple[List[Dict[str, str]], list[str]]:
        results = self.qdrant_client.search(collection_name=config.COLLECTION_NAME, query_vector=q_embedding, limit=config.SEARCH_LIMIT, with_payload=True)
        if not results: return [], []
        context_chunks = [{"text": res.payload['text'], "file": res.payload['source_file']} for res in results]
        sources = sorted(list(set([res.payload['source_file'] for res in results])))
        self._log_completion(f"найдено {len(context_chunks)} фрагментов из {len(sources)} источников")
        return context_chunks, sources
    def _log_step(self, step_num: int, message: str): print(f"\n{step_num}. {message}")
    def _log_completion(self, message: str): print(f"   ...{message}.")

    # --- ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ - ПЕРЕПИСАНА С ИСПОЛЬЗОВАНИЕМ YIELD ---
    def process_query_for_gradio(self, question: str) -> Iterator[Tuple]:
        # Начальные состояния
        blank_ds = gr.Dataset(samples=[])
        error_details = "*Ошибка*"
        
        if not question:
            yield blank_ds, blank_ds, [], "*Введите вопрос...*", None
            return

        # Шаг 1: Эмбеддинг
        self._log_step(1, "Получение эмбеддинга...")
        q_embedding = self.get_embedding(question)
        if not q_embedding:
            self._log_completion("ОШИБКА")
            yield gr.Dataset(samples=[["Ошибка: не удалось получить вектор."]]), blank_ds, [], error_details, None
            return
        self._log_completion("эмбеддинг получен")

        # Шаг 2: Поиск в Qdrant
        self._log_step(2, "Поиск релевантного контекста...")
        context_chunks, sources = self._search_and_prepare_context(q_embedding)
        sources_data = [[source] for source in sources]
        
        # --- ПЕРВАЯ ОТДАЧА РЕЗУЛЬТАТА ---
        # Сразу показываем пользователю найденные источники и статус "Генерация..."
        yield (
            gr.Dataset(samples=[["*Генерация ответа...*"]]),
            gr.Dataset(samples=sources_data),
            [],
            "*Выберите источник для просмотра...*",
            None
        )

        if not context_chunks:
            self._log_completion("контекст не найден")
            yield gr.Dataset(samples=[["Контекст не найден."]]), gr.Dataset(samples=sources_data), [], "*Контекст не найден*", None
            return

        # Шаг 3: Долгий запрос к LLM
        self._log_step(3, "Отправка запроса на LLM-сервис...")
        structured_answer = self.query_llm(question, context_chunks)
        if not structured_answer:
            self._log_completion("ОШИБКА")
            yield gr.Dataset(samples=[["Ошибка: LLM не сгенерировал ответ."]]), gr.Dataset(samples=sources_data), [], error_details, None
            return
        self._log_completion("ответ от LLM получен")
        
        answer_data = [[item['paragraph']] for item in structured_answer]
        
        # --- ВТОРАЯ И ФИНАЛЬНАЯ ОТДАЧА РЕЗУЛЬТАТА ---
        # Обновляем интерфейс, добавляя сгенерированный ответ
        yield (
            gr.Dataset(samples=answer_data),
            gr.Dataset(samples=sources_data),
            structured_answer,
            "*Выберите абзац из ответа для просмотра источника*",
            None
        )

# --- Инициализация и запуск Gradio ---
if __name__ == "__main__":
    try:
        print("Подключение к Qdrant...")
        q_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        orchestrator = RAGOrchestrator(qdrant_client=q_client)

        print("\nЗапуск интерфейса Gradio...")
        with gr.Blocks(theme=gr.themes.Soft()) as iface:
            full_response_state = gr.State([])
            gr.Markdown("...") # Заголовок

            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(lines=4, label="Ваш вопрос", placeholder="Например: Каков порядок согласования командировки?")
                    submit_btn = gr.Button("Отправить", variant="primary")
                    gr.Markdown("### Детали источника")
                    source_details_box = gr.Markdown(value="*Выберите абзац из ответа или файл-источник*")
                    gr.Markdown("### Найденные источники (файлы)")
                    sources_box = gr.Dataset(components=["text"], label="Источники", headers=["Имя файла"])

                with gr.Column(scale=3):
                    gr.Markdown("### Ответ системы")
                    answer_box = gr.Dataset(headers=["Абзац ответа"], label="Сгенерированный ответ")
                    file_preview = gr.HTML(label="Превью документа")

            submit_btn.click(
                fn=orchestrator.process_query_for_gradio,
                inputs=question_box,
                outputs=[answer_box, sources_box, full_response_state, source_details_box, file_preview]
            )
            
            answer_box.select(fn=show_source_details_from_state, inputs=[full_response_state], outputs=source_details_box)
            sources_box.select(fn=get_file_preview, inputs=None, outputs=file_preview)

        iface.launch(server_name="0.0.0.0", server_port=7860)

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ОРКЕСТРАТОРА: {e}")