import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'rules.json')

def load_rules():
    with open(RULES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def check_rules(patient):
    rules = load_rules()
    
    # 1. Критическая проверка
    if rules['critical_rules']['must_not_be_contagious'] and patient['is_contagious']:
        return "⛔️: Пациент заразен, немедленно изолировать"
    
    # 2. Проверка температуры
    if patient['temperature'] < rules['thresholds']['min_temperature']:
        return "❌: Температура ниже нормы"
    if patient['temperature'] > rules['thresholds']['max_temperature']:
        return "❌: Температура выше нормы"
    
    # 3. Проверка возраста
    if patient['age'] < rules['thresholds']['min_age'] or patient['age'] > rules['thresholds']['max_age']:
        return "❌: Возраст вне допустимого диапазона"
    
    # 4. Проверка симптомов
    for symptom in patient['symptoms']:
        if symptom in rules['lists']['danger_symptoms']:
            return f"⚠️ Найден опасный симптом ({symptom})"
    
    # Если все проверки пройдены
    return f"✅ Пациент соответствует сценарию '{rules['scenario_name']}'"
