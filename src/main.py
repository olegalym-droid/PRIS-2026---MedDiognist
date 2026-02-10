import streamlit as st

st.title("Привет, Медицинский ассистент ")
st.write("Если видите этот текст — Streamlit работает!")

if st.button("Нажми меня"):
    st.success("Все системы в норме!")
