FROM python:3.12-slim

WORKDIR /app

# Сначала копируем зависимости — это позволяет Docker кэшировать их установку
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копируем исходный код и необходимые файлы проекта
COPY . .

# Порт FastAPI
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]