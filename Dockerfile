# 1. Используем официальный легковесный образ Python
FROM mirror.gcr.io/library/python:3.11-slim-bullseye

# 2. Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# 3. Копируем файл с зависимостями
COPY requirements.txt .

# 4. Устанавливаем зависимости, не сохраняя кэш для уменьшения размера образа
RUN pip install --no-cache-dir -r requirements.txt

# 5. Копируем все остальные файлы проекта (main_app.py, config.py) в контейнер
COPY . .

# 6. Сообщаем Docker, что контейнер будет слушать на порту 7860 (стандартный для Gradio)
EXPOSE 7860

# 7. Команда для запуска приложения при старте контейнера
CMD ["python", "main_app.py"]
