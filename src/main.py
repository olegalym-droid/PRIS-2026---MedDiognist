import streamlit as st
from logic import process_text_message
from knowledge_graph import load_graph

st.set_page_config(page_title="MedDiognist Chat", layout="centered")
st.title("MedDiognist Chatbot v2.0 🧠🩺")

# Загружаем граф один раз
if "graph" not in st.session_state:
    st.session_state.graph = load_graph()

# Память чата
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отрисовка истории
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ввод пользователя
if user_input := st.chat_input("Введите симптомы или заболевание..."):

    # Команда reset
    if user_input.strip().lower() == "/reset":
        st.session_state.messages = []
        st.rerun()

    # Сообщение пользователя
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Ответ бота
    bot_response = process_text_message(
        user_input,
        st.session_state.graph
    )

    # Если логика вернула reset
    if bot_response == "__RESET__":
        st.session_state.messages = []
        st.rerun()

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response
    })

    with st.chat_message("assistant"):
        st.markdown(bot_response)
