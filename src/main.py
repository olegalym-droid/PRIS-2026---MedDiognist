import streamlit as st
from mock_data import test_patient as default_patient
from logic import check_rules

st.title("MedDiognist — Медицинский диагност 🩺")

st.write("### Настройка входящих данных пациента")

# Сайдбар для ввода данных
age = st.sidebar.number_input("Возраст пациента:", value=default_patient["age"])
temperature = st.sidebar.number_input("Температура:", value=default_patient["temperature"])
is_contagious = st.sidebar.checkbox("Пациент заразен?", value=default_patient["is_contagious"])
symptoms_input = st.sidebar.text_area(
    "Симптомы:", 
    value=", ".join(default_patient["symptoms"])
)
symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

if st.button("Запустить проверку"):
    patient_data = {
        "age": age,
        "temperature": temperature,
        "is_contagious": is_contagious,
        "symptoms": symptoms
    }
    
    result = check_rules(patient_data)
    
    if "✅" in result:
        st.success(result)
    elif "⛔️" in result:
        st.error(result)
    else:
        st.warning(result)
