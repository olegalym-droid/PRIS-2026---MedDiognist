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

def normalize_text(text):
    return text.lower().strip()


def process_text_message(text, graph):
    text = normalize_text(text)

    # 🔹 Команда сброса
    if text == "/reset":
        return "__RESET__"

    # 🔹 Приветствие
    if "привет" in text:
        return "Здраствуйте! Я MedDiognist 🩺 Напиши симптомы или название болезни."

    # 🔹 Разбиваем на слова
    words = text.replace(",", " ").split()

    found_nodes = []
    for word in words:
        if word in graph.nodes:
            found_nodes.append(word)

    # 🔹 Если нашли термины
    if found_nodes:
        response = []

        for node in found_nodes:
            neighbors = list(graph.neighbors(node))

            if neighbors:
                response.append(
                    f"🔎 '{node}' связано с: {', '.join(neighbors)}"
                )
            else:
                response.append(f"🔎 '{node}' найдено, но связей нет.")

        return "\n\n".join(response)

    # 🔹 Попытка автодиагностики по симптомам
    symptoms = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "symptom" and node in words:
            symptoms.append(node)

    if len(symptoms) >= 2:
        possible_diseases = []

        for node, data in graph.nodes(data=True):
            if data.get("type") == "disease":
                disease_symptoms = list(graph.neighbors(node))
                matches = set(symptoms) & set(disease_symptoms)

                if len(matches) >= 2:
                    possible_diseases.append(node)

        if possible_diseases:
            return f"🩺 Возможные заболевания: {', '.join(possible_diseases)}"

    return "Я не знаю такого термина. Попробуйте ввести симптом или болезнь."
