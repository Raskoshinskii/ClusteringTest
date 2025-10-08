#!/bin/bash

# Скрипт для запуска Streamlit приложения иерархической кластеризации

echo "🚀 Запуск приложения иерархической кластеризации..."
echo "📂 Текущая директория: $(pwd)"

# Проверяем наличие app.py
if [ ! -f "app.py" ]; then
    echo "❌ Файл app.py не найден в текущей директории"
    exit 1
fi

# Проверяем наличие requirements.txt и устанавливаем зависимости
if [ -f "requirements.txt" ]; then
    echo "📦 Проверка зависимостей..."
    pip install -r requirements.txt
else
    echo "⚠️  Файл requirements.txt не найден"
fi

# Запуск Streamlit
echo "🌐 Запуск Streamlit сервера..."
echo "📍 Приложение будет доступно по адресу: http://localhost:8502"
echo "🔄 Для остановки нажмите Ctrl+C"

streamlit run app.py --server.port 8502 --server.address localhost