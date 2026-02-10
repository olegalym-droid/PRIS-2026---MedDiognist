# MedDiognist

## Title
MedDiognist — учебный медицинский помощник

## Description
MedDiognist — это учебный проект, разработанный в рамках университетского курса.  
Проект предназначен для демонстрации архитектуры интеллектуальной системы, которая принимает симптомы от пользователя и возвращает предварительную информацию о возможном диагнозе.

## Tech Stack
В проекте планируется использование следующих технологий:

- Python 3
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- GitHub

## Architecture Diagram

```mermaid
graph TD;
    User[Пользователь] -->|Ввод симптомов| UI[Streamlit интерфейс];
    UI -->|Данные| Logic[Логика обработки];
    Logic -->|Признаки| Model[AI модель];
    Model -->|Результат| UI;
    UI -->|Ответ| User;
    
PRIS-2026---MedDiognist/
├── .vscode/
│   └── settings.json        # Настройки VS Code
├── data/
│   ├── raw/                 # Исходные данные
│   └── processed/           # Обработанные данные
├── docs/                    # Документация и схемы
├── notebooks/               # Jupyter ноутбуки для экспериментов
├── src/
│   ├── __init__.py
│   └── main.py              # Точка входа приложения
├── .gitignore               # Исключения для Git
├── README.md                # Описание проекта
└── requirements.txt         # Зависимости
