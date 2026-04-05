# src/main.py

import streamlit as st

from knowledge_graph import DISPLAY_RU, load_graph
from logic import analyze_case, result_to_text, triage_label

TEXT = {
    "ru": {
        "app_title": "MedDiognist",
        "sidebar_settings": "Настройки",
        "language": "Язык",
        "examples": "Примеры",
        "state": "Состояние",
        "confirmed": "Подтверждено",
        "denied": "Отрицано",
        "pending": "Уточнение",
        "triage": "Риск",
        "reset": "Сбросить",
        "chat": "Чат",
        "analysis": "Текущий анализ",
        "chat_placeholder": "Опишите симптомы или введите /reset",
        "risk_level": "Уровень риска",
        "recognized_symptoms": "Распознанные симптомы",
        "suggested_specialty": "Рекомендуемый врач",
        "top_cases": "Подходящие клинические сценарии",
        "next_question": "Следующий вопрос",
        "summary": "Предварительный вывод",
        "final_recommendation": "Что делать сейчас",
        "start_hint": "Начните с ввода симптомов.",
        "matched": "Совпадения",
        "disclaimer": "Учебный symptom-checker. Не является медицинским диагнозом.",
        "example_1": "кашель, температура, боль в горле",
        "example_2": "боль в груди и одышка",
        "example_3": "жажда, частое мочеиспускание, размытое зрение",
        "example_4": "потеря памяти, забывчивость, спутанность сознания",
        "example_5": "тошнота, боль в животе, пожелтение кожи",
    },
    "en": {
        "app_title": "MedDiognist",
        "sidebar_settings": "Settings",
        "language": "Language",
        "examples": "Examples",
        "state": "State",
        "confirmed": "Confirmed",
        "denied": "Denied",
        "pending": "Pending",
        "triage": "Risk",
        "reset": "Reset",
        "chat": "Chat",
        "analysis": "Current analysis",
        "chat_placeholder": "Describe symptoms or type /reset",
        "risk_level": "Risk level",
        "recognized_symptoms": "Recognized symptoms",
        "suggested_specialty": "Recommended doctor",
        "top_cases": "Relevant clinical scenarios",
        "next_question": "Next question",
        "summary": "Preliminary conclusion",
        "final_recommendation": "What to do now",
        "start_hint": "Start by entering symptoms.",
        "matched": "Matches",
        "disclaimer": "Educational symptom-checker. Not a medical diagnosis.",
        "example_1": "cough, fever, sore throat",
        "example_2": "chest pain and shortness of breath",
        "example_3": "thirst, frequent urination, blurred vision",
        "example_4": "memory loss, forgetfulness, confusion",
        "example_5": "nausea, abdominal pain, jaundice",
    },
}

DEFAULT_PATIENT_DATA = {
    "confirmed": [],
    "denied": [],
    "pending_question": None,
    "emergency_triggered": False,
    "emergency_term": None,
    "triage": None,
}

def t(key: str) -> str:
    return TEXT[st.session_state.lang][key]


def init_state():
    if "graph" not in st.session_state:
        st.session_state.graph = load_graph()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    if "lang" not in st.session_state:
        st.session_state.lang = "ru"
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def reset_app():
    st.session_state.messages = []
    st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    st.session_state.prefill = ""
    st.session_state.last_result = None


def render_state_symptoms(values):
    if not values:
        return "—"
    if st.session_state.lang == "en":
        return ", ".join(values)
    translated = [DISPLAY_RU.get(value, value) for value in values]
    return ", ".join(translated)


st.set_page_config(page_title="MedDiognist", layout="wide")
init_state()

st.title(t("app_title"))
st.caption(t("disclaimer"))

with st.sidebar:
    st.markdown(f"### {t('sidebar_settings')}")
    st.session_state.lang = st.radio(
        t("language"),
        ["ru", "en"],
        index=0 if st.session_state.lang == "ru" else 1,
        horizontal=True,
    )

    st.markdown(f"### {t('examples')}")
    for key in ["example_1", "example_2", "example_3", "example_4", "example_5"]:
        if st.button(t(key), use_container_width=True):
            st.session_state.prefill = t(key)

    st.markdown(f"### {t('state')}")
    st.write(f"**{t('confirmed')}:**", render_state_symptoms(st.session_state.patient_data["confirmed"]))
    st.write(f"**{t('denied')}:**", render_state_symptoms(st.session_state.patient_data["denied"]))
    pending = st.session_state.patient_data["pending_question"]
    pending_display = DISPLAY_RU.get(pending, pending) if pending and st.session_state.lang == "ru" else pending
    st.write(f"**{t('pending')}:**", pending_display or "—")
    st.write(
        f"**{t('triage')}:**",
        triage_label(st.session_state.patient_data["triage"], st.session_state.lang)
        if st.session_state.patient_data["triage"]
        else "—",
    )

    if st.button(t("reset"), use_container_width=True):
        reset_app()
        st.rerun()

left, right = st.columns([1.15, 1])

with left:
    st.subheader(t("chat"))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(t("chat_placeholder"))

    if not user_input and st.session_state.prefill:
        user_input = st.session_state.prefill
        st.session_state.prefill = ""

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = analyze_case(
            user_input,
            st.session_state.graph,
            st.session_state.patient_data,
            lang=st.session_state.lang,
            mutate=True,
        )
        bot_response = result_to_text(result, st.session_state.graph, st.session_state.lang)

        if bot_response == "__RESET__":
            reset_app()
            st.rerun()

        st.session_state.last_result = result
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        with st.chat_message("assistant"):
            st.markdown(bot_response)

with right:
    st.subheader(t("analysis"))
    result = st.session_state.last_result

    if result and result.get("triage"):
        st.metric(t("risk_level"), triage_label(result["triage"], st.session_state.lang))

    if result and result.get("summary"):
        st.markdown(f"#### {t('summary')}")
        st.write(result["summary"])

    if result and result.get("final_recommendation"):
        st.markdown(f"#### {t('final_recommendation')}")
        st.success(result["final_recommendation"])

    if result and result.get("recognized_symptoms"):
        st.markdown(f"#### {t('recognized_symptoms')}")
        for symptom in result["recognized_symptoms"]:
            st.write(f"- {symptom}")

    if result and result.get("specialties"):
        st.markdown(f"#### {t('suggested_specialty')}")
        for item in result["specialties"]:
            st.write(f"- {item['name']}")

    if result and result.get("cases"):
        st.markdown(f"#### {t('top_cases')}")
        for item in result["cases"]:
            title = item["case"]
            if item["specialty"]:
                title += f" ({item['specialty']})"
            with st.expander(f"{title} — {item['score']}%"):
                if item["matches"]:
                    st.write(f"{t('matched')}:")
                    for match in item["matches"]:
                        st.write(f"- {match}")

    if result and result.get("question"):
        st.markdown(f"#### {t('next_question')}")
        st.write(result["question"])

    if not result:
        st.info(t("start_hint"))