import streamlit as st
from knowledge_graph import load_graph
from logic import process_text_message

st.set_page_config(page_title="MedDiognist Chat", layout="centered")
st.title("MedDiognist Chatbot 🧠🩺")

if "graph" not in st.session_state:
    st.session_state.graph = load_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "confirmed": [],
        "denied": [],
        "pending_question": None,
        "emergency_triggered": False,
        "emergency_symptom": None,
    }

with st.sidebar:
    st.markdown("### Текущее состояние")
    st.write("✅ Подтверждено:", ", ".join(st.session_state.patient_data["confirmed"]) or "—")
    st.write("❌ Отрицано:", ", ".join(st.session_state.patient_data["denied"]) or "—")
    st.write("❓ Вопрос:", st.session_state.patient_data["pending_question"] or "—")
    if st.button("Сбросить (/reset)"):
        st.session_state.messages = []
        st.session_state.patient_data = {
            "confirmed": [],
            "denied": [],
            "pending_question": None,
            "emergency_triggered": False,
            "emergency_symptom": None,
        }
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Опишите симптомы... (или /reset)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    bot_response = process_text_message(user_input, st.session_state.graph, st.session_state.patient_data)

    if bot_response == "__RESET__":
        st.session_state.messages = []
        st.session_state.patient_data = {
            "confirmed": [],
            "denied": [],
            "pending_question": None,
            "emergency_triggered": False,
            "emergency_symptom": None,
        }
        st.rerun()

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)