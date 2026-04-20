import streamlit as st

from knowledge_graph import DISPLAY_RU, load_graph
from logic import analyze_case, triage_label

try:
    from similarity import CaseSimilarity
except Exception:
    CaseSimilarity = None

TEXT = {
    "ru": {
        "app_title": "MedDiognist",
        "subtitle": "Учебный медицинский помощник для предварительной оценки симптомов",
        "sidebar_settings": "Настройки",
        "language": "Язык",
        "examples": "Примеры",
        "state": "Состояние диалога",
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
        "pipeline": "Как система приняла решение",
        "pipeline_step_1": "1. Получен текст жалобы",
        "pipeline_step_2": "2. Выделены симптомы",
        "pipeline_step_3": "3. Оценен риск",
        "pipeline_step_4": "4. Подобраны сценарии и специальность",
        "pipeline_step_5": "5. Сформирована рекомендация",
        "raw_input": "Входной текст",
        "no_data": "Пока нет данных",
        "reasoning": "Обоснование результата",
        "reasoning_risk": "Почему такой риск",
        "reasoning_spec": "Почему этот специалист",
        "reasoning_cases": "Почему эти сценарии",
        "emergency_alert": "Обнаружен опасный симптом. Нужна срочная медицинская помощь.",
        "chat_welcome": (
            "Здравствуйте! Опишите симптомы свободным текстом, "
            "а я попробую определить уровень риска, возможное направление консультации и дальнейшее действие."
        ),
        "similar_cases": "Похожие случаи по текстовому описанию",
        "similar_cases_hint": "Этот блок использует TF-IDF и cosine similarity для поиска похожих описаний случаев в датасете.",
        "similarity_unavailable": "Модуль similarity пока недоступен. Проверьте установку scikit-learn.",
        "similarity_empty": "Для поиска похожих случаев сначала введите жалобу.",
        "similarity_score": "Сходство",
    },
    "en": {
        "app_title": "MedDiognist",
        "subtitle": "Educational medical assistant for preliminary symptom assessment",
        "sidebar_settings": "Settings",
        "language": "Language",
        "examples": "Examples",
        "state": "Dialogue state",
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
        "pipeline": "How the system made the decision",
        "pipeline_step_1": "1. Complaint text received",
        "pipeline_step_2": "2. Symptoms extracted",
        "pipeline_step_3": "3. Risk estimated",
        "pipeline_step_4": "4. Scenarios and specialty selected",
        "pipeline_step_5": "5. Recommendation generated",
        "raw_input": "Input text",
        "no_data": "No data yet",
        "reasoning": "Reasoning",
        "reasoning_risk": "Why this risk level",
        "reasoning_spec": "Why this specialist",
        "reasoning_cases": "Why these scenarios",
        "emergency_alert": "A dangerous symptom was detected. Urgent medical care is needed.",
        "chat_welcome": (
            "Hello! Describe symptoms in free text, "
            "and I will try to estimate the risk level, suggest a consultation direction, and recommend what to do next."
        ),
        "similar_cases": "Similar cases by text description",
        "similar_cases_hint": "This block uses TF-IDF and cosine similarity to retrieve similar case descriptions from the dataset.",
        "similarity_unavailable": "Similarity module is unavailable. Check whether scikit-learn is installed.",
        "similarity_empty": "Enter a complaint first to search for similar cases.",
        "similarity_score": "Similarity",
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
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": TEXT["ru"]["chat_welcome"],
            }
        ]
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    if "lang" not in st.session_state:
        st.session_state.lang = "ru"
    if "prefill" not in st.session_state:
        st.session_state.prefill = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = ""
    if "similarity_model" not in st.session_state:
        st.session_state.similarity_model = None
        st.session_state.similarity_error = None
        if CaseSimilarity is not None:
            try:
                st.session_state.similarity_model = CaseSimilarity()
            except Exception as e:
                st.session_state.similarity_error = str(e)
        else:
            st.session_state.similarity_error = "CaseSimilarity import failed"
    if "similar_cases" not in st.session_state:
        st.session_state.similar_cases = []


def reset_app():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": TEXT[st.session_state.lang]["chat_welcome"],
        }
    ]
    st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    st.session_state.prefill = ""
    st.session_state.last_result = None
    st.session_state.last_user_input = ""
    st.session_state.similar_cases = []


def render_state_symptoms(values):
    if not values:
        return "—"
    if st.session_state.lang == "en":
        return ", ".join(values)
    translated = [DISPLAY_RU.get(value, value) for value in values]
    return ", ".join(translated)


def render_message_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def build_bot_response(result: dict) -> str:
    lang = st.session_state.lang

    if result.get("action") == "reset":
        return "__RESET__"

    if result.get("action") == "need_input":
        return result.get("message", "")

    if result.get("action") == "emergency_locked":
        return result.get("message", "")

    if result.get("action") == "emergency":
        lines = []
        if lang == "ru":
            lines.append(f"**Уровень риска:** {triage_label(result['triage'], lang)}")
            lines.append(f"**Опасный симптом:** {result['recognized_symptoms'][-1] if result.get('recognized_symptoms') else '—'}")
        else:
            lines.append(f"**Risk level:** {triage_label(result['triage'], lang)}")
            lines.append(f"**Dangerous symptom:** {result['recognized_symptoms'][-1] if result.get('recognized_symptoms') else '—'}")

        if result.get("final_recommendation"):
            lines.append(result["final_recommendation"])
        return "\n\n".join(lines)

    lines = []

    if result.get("triage"):
        if lang == "ru":
            lines.append(f"**Уровень риска:** {triage_label(result['triage'], lang)}")
        else:
            lines.append(f"**Risk level:** {triage_label(result['triage'], lang)}")

    if result.get("summary"):
        lines.append(result["summary"])

    if result.get("final_recommendation"):
        lines.append(result["final_recommendation"])

    if result.get("recognized_symptoms"):
        if lang == "ru":
            lines.append("**Распознанные симптомы:** " + ", ".join(result["recognized_symptoms"]))
        else:
            lines.append("**Recognized symptoms:** " + ", ".join(result["recognized_symptoms"]))

    if result.get("question"):
        if lang == "ru":
            lines.append(f"**Уточняющий вопрос:** есть ли у вас {result['question']}?")
        else:
            lines.append(f"**Clarifying question:** do you have {result['question']}?")

    return "\n\n".join(lines)


def render_pipeline(result: dict):
    st.markdown(f"#### {t('pipeline')}")

    raw_input = st.session_state.last_user_input.strip() or "—"
    st.markdown(f"**{t('pipeline_step_1')}**")
    st.write(f"{t('raw_input')}: {raw_input}")

    st.markdown(f"**{t('pipeline_step_2')}**")
    symptoms = result.get("recognized_symptoms") or []
    if symptoms:
        for symptom in symptoms:
            st.write(f"- {symptom}")
    else:
        st.write(t("no_data"))

    st.markdown(f"**{t('pipeline_step_3')}**")
    if result.get("triage"):
        st.write(triage_label(result["triage"], st.session_state.lang))
    else:
        st.write(t("no_data"))

    st.markdown(f"**{t('pipeline_step_4')}**")
    has_any = False
    for item in result.get("specialties", []):
        has_any = True
        st.write(f"- {item['name']}")
    for item in result.get("cases", []):
        has_any = True
        st.write(f"- {item['case']} — {item['score']}%")
    if not has_any:
        st.write(t("no_data"))

    st.markdown(f"**{t('pipeline_step_5')}**")
    if result.get("final_recommendation"):
        st.write(result["final_recommendation"])
    else:
        st.write(t("no_data"))


def render_reasoning(result: dict):
    st.markdown(f"#### {t('reasoning')}")

    with st.expander(t("reasoning_risk"), expanded=True):
        triage = result.get("triage")
        symptoms = result.get("recognized_symptoms") or []

        if not triage:
            st.write(t("no_data"))
        else:
            if st.session_state.lang == "ru":
                st.write(f"Система определила уровень риска: **{triage_label(triage, 'ru')}**.")
                if symptoms:
                    st.write("На решение повлияли распознанные симптомы:")
                    for symptom in symptoms:
                        st.write(f"- {symptom}")
            else:
                st.write(f"The system estimated the risk as **{triage_label(triage, 'en')}**.")
                if symptoms:
                    st.write("Recognized symptoms that affected the decision:")
                    for symptom in symptoms:
                        st.write(f"- {symptom}")

    with st.expander(t("reasoning_spec"), expanded=True):
        specialties = result.get("specialties", [])
        if not specialties:
            st.write(t("no_data"))
        else:
            if st.session_state.lang == "ru":
                st.write("Специалист выбран на основе набора симптомов и их соответствия предметной области.")
            else:
                st.write("The specialist is selected based on the symptom set and domain relevance.")
            for item in specialties:
                st.write(f"- {item['name']}")

    with st.expander(t("reasoning_cases"), expanded=False):
        cases = result.get("cases", [])
        if not cases:
            st.write(t("no_data"))
        else:
            if st.session_state.lang == "ru":
                st.write("Клинические сценарии ранжируются по совпадению симптомов и внутреннему score.")
            else:
                st.write("Clinical scenarios are ranked by symptom overlap and internal score.")
            for item in cases:
                st.markdown(f"**{item['case']}** — {item['score']}%")
                if item.get("specialty"):
                    st.write(item["specialty"])
                if item.get("matches"):
                    for match in item["matches"]:
                        st.write(f"- {match}")


def update_similar_cases():
    text = st.session_state.last_user_input.strip()
    model = st.session_state.similarity_model

    if not text or model is None:
        st.session_state.similar_cases = []
        return

    try:
        st.session_state.similar_cases = model.find_similar(text, top_k=3)
    except Exception:
        st.session_state.similar_cases = []


def render_similar_cases():
    st.markdown(f"#### {t('similar_cases')}")
    st.caption(t("similar_cases_hint"))

    if st.session_state.similarity_model is None:
        st.info(t("similarity_unavailable"))
        return

    if not st.session_state.last_user_input.strip():
        st.info(t("similarity_empty"))
        return

    similar_cases = st.session_state.similar_cases or []
    if not similar_cases:
        st.write(t("no_data"))
        return

    for item in similar_cases:
        with st.expander(f"{item['case']} — {item['score']}%"):
            st.write(f"{t('similarity_score')}: {item['score']}%")


st.set_page_config(page_title="MedDiognist", layout="wide")
init_state()

st.title(t("app_title"))
st.caption(t("subtitle"))
st.caption(t("disclaimer"))

with st.sidebar:
    st.markdown(f"### {t('sidebar_settings')}")
    previous_lang = st.session_state.lang
    st.session_state.lang = st.radio(
        t("language"),
        ["ru", "en"],
        index=0 if st.session_state.lang == "ru" else 1,
        horizontal=True,
    )

    if previous_lang != st.session_state.lang and st.session_state.messages:
        if st.session_state.messages[0]["role"] == "assistant":
            st.session_state.messages[0]["content"] = TEXT[st.session_state.lang]["chat_welcome"]

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

    current_triage = st.session_state.patient_data["triage"]
    st.write(
        f"**{t('triage')}:**",
        triage_label(current_triage, st.session_state.lang) if current_triage else "—",
    )

    if st.button(t("reset"), use_container_width=True):
        reset_app()
        st.rerun()

left, right = st.columns([1.15, 1])

with left:
    st.subheader(t("chat"))
    render_message_history()

    user_input = st.chat_input(t("chat_placeholder"))

    if not user_input and st.session_state.prefill:
        user_input = st.session_state.prefill
        st.session_state.prefill = ""

    if user_input:
        st.session_state.last_user_input = user_input
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

        update_similar_cases()

        bot_response = build_bot_response(result)

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

    if result and result.get("action") == "emergency":
        st.error(t("emergency_alert"))

    if result and result.get("triage"):
        st.metric(t("risk_level"), triage_label(result["triage"], st.session_state.lang))

    if result and result.get("summary"):
        st.markdown(f"#### {t('summary')}")
        st.write(result["summary"])

    if result and result.get("final_recommendation"):
        st.markdown(f"#### {t('final_recommendation')}")
        if result.get("triage") == "CRITICAL":
            st.error(result["final_recommendation"])
        elif result.get("triage") == "HIGH":
            st.warning(result["final_recommendation"])
        else:
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

    render_similar_cases()

    if result:
        render_pipeline(result)
        render_reasoning(result)
    else:
        st.info(t("start_hint"))