import requests
import gradio as gr
import config
from qdrant_client import QdrantClient
from typing import Optional, Tuple, List
from bs4 import BeautifulSoup
from urllib.parse import quote

def get_file_preview(evt: gr.SelectData):
    """
    Обработчик клика по 'скрепке'. Формирует URL для скачивания файла 
    с сервиса rag-client.
    """
    try:
        soup = BeautifulSoup(evt.value, "html.parser")
        link = soup.find('a')
        if link and link.has_attr('href'):
            file_ref = link['href'].split('=')[-1]
            
            # Кодируем имя файла для безопасной передачи в URL
            encoded_file_ref = quote(file_ref)
            
            # Формируем полный URL для доступа к файлу
            file_url = f"{config.DOCS_ENDPOINT.strip('/')}/{encoded_file_ref}"
            print(f"Запрос превью для URL: {file_url}")
            return file_url
    except Exception as e:
        print(f"Ошибка при обработке клика для превью: {e}")
    
    return None

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

    def process_query(self, question: str) -> Tuple[str, str, object]:
        """Полный цикл обработки вопроса от пользователя."""
        if not question:
            return "Пожалуйста, введите вопрос.", "", None

        self._log_step(1, f"Получение эмбеддинга для вопроса: '{question[:30]}...'")
        question_embedding = self.get_embedding(question)
        if not question_embedding:
            return "Не удалось получить вектор для вопроса. Проверьте сервис эмбеддингов.", ""
        self._log_completion("эмбеддинг получен")

        self._log_step(2, "Поиск релевантного контекста в Qdrant...")
        context, sources = self._search_and_prepare_context(question_embedding)
        if not context:
            return "В базе знаний не найдено релевантного контекста.", "", None

        self._log_step(3, "Отправка запроса на LLM-сервис...")
        answer = self.query_llm(question, context)
        self._log_completion("ответ от LLM получен")

        answer_with_clips = self._add_paperclips_to_answer(answer, sources)
        
        return answer_with_clips, f"Источники: {', '.join(sources)}", None
    
    def _add_paperclips_to_answer(self, answer: str, sources: List[str]) -> str:    
        """Добавляет иконки-скрепки к абзацам ответа."""
        # Простой вариант: добавляем скрепки со ссылкой на первый источник
        # В более сложном варианте нужно сопоставлять абзац с его источником
        if not sources:
            return answer

        # Используем BeautifulSoup для безопасной работы с HTML-подобной структурой
        soup = BeautifulSoup(f"<div>{answer}</div>", "html.parser")
        paragraphs = soup.find_all('p') # LLM часто возвращает <p> теги
        if not paragraphs:
             # Если <p> нет, просто работаем с текстом
             paragraphs = soup.get_text().split('\n')

        for i, p in enumerate(paragraphs):
            source_file = sources[i % len(sources)] # Циклически берем источники
            clip_html = f'<a href="file={source_file}" class="paperclip" title="Показать {source_file}">📎</a>'
            if isinstance(p, str):
                 paragraphs[i] = f"<p>{p} {clip_html}</p>"
            else:
                 p.append(BeautifulSoup(clip_html, "html.parser"))


        return "".join(map(str, paragraphs))
    
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
        with gr.Blocks() as iface:
            gr.Markdown(
                """
                # RAG-система для ВНД Атомстройкомплекс
                Введите свой вопрос. Система найдет релевантные документы и сгенерирует ответ.
                Кликните на 📎 для просмотра исходного документа.
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(lines=3, label="Ваш вопрос к базе знаний")
                    submit_btn = gr.Button("Отправить")
                    sources_box = gr.Textbox(label="Найденные источники")
                with gr.Column(scale=3):
                    answer_box = gr.HTML(label="Ответ", elem_id="answer_display") # Используем HTML для отображения скрепок
                    file_preview = gr.File(label="Превью документа")

            submit_btn.click(
                fn=orchestrator.process_query,
                inputs=question_box,
                outputs=[answer_box, sources_box, file_preview]
            )

            # Добавляем JavaScript для обработки кликов по скрепкам
            answer_box.change(
                None,
                None,
                None,
                js="""
                function() {
                    setTimeout(() => {
                        document.querySelectorAll('#answer_display a[href^="file="]').forEach(link => {
                            link.onclick = function(e) {
                                e.preventDefault();
                                const fileName = this.href.split('file=')[1];
                                console.log('Клик по файлу:', fileName);
                                // Здесь можно добавить логику для загрузки файла
                            };
                        });
                    }, 100);
                }
                """
            )

        # Запускаем Gradio на порту 80, чтобы был доступен по IP машины
        iface.launch(server_name="0.0.0.0", server_port=7860)

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ ОРКЕСТРАТОРА: {e}")