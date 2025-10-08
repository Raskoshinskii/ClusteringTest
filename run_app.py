#!/usr/bin/env python3
"""
Скрипт для запуска Streamlit приложения иерархической кластеризации
"""

import subprocess
import sys
import webbrowser
import time
import os

def run_streamlit_app():
    """Запуск Streamlit приложения"""
    try:
        print("🚀 Запуск Streamlit приложения...")
        print("📂 Рабочая директория:", os.getcwd())
        
        # Запуск Streamlit
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
        print("✅ Приложение запущено!")
        print("🌐 Откройте браузер и перейдите по адресу: http://localhost:8501")
        
        # Пауза для запуска сервера
        time.sleep(3)
        
        # Автоматическое открытие браузера
        try:
            webbrowser.open("http://localhost:8501")
        except:
            print("❌ Не удалось автоматически открыть браузер")
        
        # Ожидание завершения процесса
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Приложение остановлено пользователем")
        process.terminate()
    except Exception as e:
        print(f"❌ Ошибка при запуске: {e}")
        print("💡 Попробуйте запустить вручную: streamlit run app.py")

if __name__ == "__main__":
    run_streamlit_app()