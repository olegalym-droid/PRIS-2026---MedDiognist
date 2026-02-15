import streamlit as st
from knowledge_graph import create_graph, find_diseases_by_symptoms
from logic import check_rules

st.title("MedDiognist — Intelligent System 🧠🩺")

# Загружаем граф
G = create_graph()

st.write("### Введите данные пациента")

age = st.number_input("Возраст:", min_value=0, max_value=120, value=30)
temperature = st.number_input("Температура:", value=37.0)
is_contagious = st.checkbox("Пациент заразен?")
symptoms_input = st.text_area("Симптомы (через запятую):")

input_symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

if st.button("Анализировать"):

    patient_data = {
        "age": age,
        "temperature": temperature,
        "is_contagious": is_contagious,
        "symptoms": input_symptoms
    }


    rule_result = check_rules(patient_data)

    if "⛔️" in rule_result:
        st.error(rule_result)
    else:
        st.info(rule_result)


        diseases = find_diseases_by_symptoms(G, input_symptoms)

        if diseases:
            st.success(f"Возможные заболевания: {', '.join(diseases)}")

            for disease in diseases:
                meds = [
                    n for n in G.neighbors(disease)
                    if G.nodes[n]["type"] == "medicine"
                ]
                st.write(f"💊 Рекомендуемые препараты для {disease}: {', '.join(meds)}")
        else:
            st.warning("Болезни по введенным симптомам не найдены.")
