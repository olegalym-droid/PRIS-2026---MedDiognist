import streamlit as st

from knowledge_graph import DISPLAY_RU, MANUAL_CASES, load_graph
from logic import analyze_case, triage_label, format_specialty_name
from ml_predictor import SpecialtyMLPredictor


TEXT = {
    "ru": {
        "app_title": "MedDiognist",
        "subtitle": "Учебный медицинский помощник для предварительной оценки симптомов",
        "disclaimer": "Учебный symptom-checker. Не является медицинским диагнозом.",
        "language": "Язык",
        "constructor": "Конструктор симптомов",
        "analyze": "Анализировать",
        "clear": "Сбросить",
        "selection": "Выбранные симптомы",
        "none": "Пока ничего не выбрано",
        "analysis": "Текущий анализ",
        "risk_level": "Уровень риска",
        "summary": "Предварительный вывод",
        "final_recommendation": "Что делать сейчас",
        "recognized_symptoms": "Распознанные симптомы",
        "specialties": "Рекомендуемый врач",
        "cases": "Подходящие клинические сценарии",
        "matched": "Совпадения",
        "pending_question": "Уточняющий вопрос",
        "answer_yes": "Да",
        "answer_no": "Нет",
        "need_selection": "Сначала выберите хотя бы один симптом.",
        "selection_sent": "Симптомы отправлены в анализ.",
        "reset_done": "Состояние сброшено.",
        "start_hint": "Отметьте симптомы слева и нажмите «Анализировать».",
        "danger_alert": "Обнаружен опасный симптом. Нужна срочная медицинская помощь.",
        "state": "Состояние",
        "confirmed": "Подтверждено",
        "denied": "Отрицано",
        "pending": "Уточнение",
        "triage": "Риск",
        "probability": "Вероятность соответствия",
        "symptom_groups": "Категории симптомов",
        "ml_title": "ML-прогноз специальности",
        "ml_best": "Наиболее вероятное направление",
        "ml_top": "Топ направлений по модели",
        "ml_probability": "Вероятность модели",
        "ml_cases": "Обучающих примеров",
        "ml_unavailable": "ML-модель пока недоступна.",
        "ml_not_enough": "Недостаточно данных для ML-прогноза.",
        "ml_case_title": "ML-прогноз клинического сценария",
        "ml_case_best": "Наиболее вероятный сценарий",
        "ml_case_top": "Топ сценариев по модели",
        "ml_features": "Симптомы, характерные для сценария",
        "ml_error_title": "Ошибка ML-модуля",
        "ml_debug": "Техническая информация",
    },
    "en": {
        "app_title": "MedDiognist",
        "subtitle": "Educational medical assistant for preliminary symptom assessment",
        "disclaimer": "Educational symptom-checker. Not a medical diagnosis.",
        "language": "Language",
        "constructor": "Symptom builder",
        "analyze": "Analyze",
        "clear": "Reset",
        "selection": "Selected symptoms",
        "none": "Nothing selected yet",
        "analysis": "Current analysis",
        "risk_level": "Risk level",
        "summary": "Preliminary conclusion",
        "final_recommendation": "What to do now",
        "recognized_symptoms": "Recognized symptoms",
        "specialties": "Recommended doctor",
        "cases": "Relevant clinical scenarios",
        "matched": "Matches",
        "pending_question": "Clarifying question",
        "answer_yes": "Yes",
        "answer_no": "No",
        "need_selection": "Select at least one symptom first.",
        "selection_sent": "Symptoms were sent to the analysis.",
        "reset_done": "State was reset.",
        "start_hint": "Select symptoms on the left and press Analyze.",
        "danger_alert": "A dangerous symptom was detected. Urgent medical care is needed.",
        "state": "State",
        "confirmed": "Confirmed",
        "denied": "Denied",
        "pending": "Pending",
        "triage": "Risk",
        "probability": "Match probability",
        "symptom_groups": "Symptom groups",
        "ml_title": "ML specialty prediction",
        "ml_best": "Most probable direction",
        "ml_top": "Top directions by model",
        "ml_probability": "Model probability",
        "ml_cases": "Training samples",
        "ml_unavailable": "ML model is unavailable.",
        "ml_not_enough": "Not enough data for ML prediction.",
        "ml_case_title": "ML clinical scenario prediction",
        "ml_case_best": "Most probable scenario",
        "ml_case_top": "Top scenarios by model",
        "ml_features": "Symptoms typical for the scenario",
        "ml_error_title": "ML module error",
        "ml_debug": "Technical details",
    },
}

CURATED_GROUPS = {
    "ru": {
        "Общие": [
            "fever",
            "chills",
            "fatigue",
            "weakness",
            "weight loss",
            "weight gain",
            "drowsiness",
        ],
        "Дыхательная система": [
            "cough",
            "dry cough",
            "productive cough",
            "shortness of breath",
            "sore throat",
            "rhinorrhea",
            "nasal congestion",
            "sneezing",
            "chest pain",
        ],
        "ЖКТ и живот": [
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "abdominal pain",
            "heartburn",
            "bloating",
            "blood in stool",
            "jaundice",
        ],
        "Неврология": [
            "headache",
            "dizziness",
            "numbness",
            "seizure",
            "loss of consciousness",
            "memory loss",
            "forgetfulness",
            "confusion",
            "tremor",
        ],
        "Мочевыделительная система": [
            "dysuria",
            "urinary frequency",
            "hematuria",
            "flank pain",
            "back pain",
        ],
        "Кожа и аллергия": [
            "rash",
            "itching",
            "erythema",
            "leg swelling",
        ],
        "Эндокринология и обмен": [
            "dry mouth",
            "excessive thirst",
            "increased appetite",
            "blurred vision",
            "vision loss",
            "palpitations",
            "insomnia",
            "anxiety",
            "depressed mood",
        ],
        "Уши и слух": [
            "ear pain",
            "tinnitus",
            "hearing loss",
        ],
        "Опорно-двигательная система": [
            "joint pain",
            "muscle pain",
        ],
    },
    "en": {
        "General": [
            "fever",
            "chills",
            "fatigue",
            "weakness",
            "weight loss",
            "weight gain",
            "drowsiness",
        ],
        "Respiratory": [
            "cough",
            "dry cough",
            "productive cough",
            "shortness of breath",
            "sore throat",
            "rhinorrhea",
            "nasal congestion",
            "sneezing",
            "chest pain",
        ],
        "GI and Abdomen": [
            "nausea",
            "vomiting",
            "diarrhea",
            "constipation",
            "abdominal pain",
            "heartburn",
            "bloating",
            "blood in stool",
            "jaundice",
        ],
        "Neurology": [
            "headache",
            "dizziness",
            "numbness",
            "seizure",
            "loss of consciousness",
            "memory loss",
            "forgetfulness",
            "confusion",
            "tremor",
        ],
        "Urinary": [
            "dysuria",
            "urinary frequency",
            "hematuria",
            "flank pain",
            "back pain",
        ],
        "Skin and Allergy": [
            "rash",
            "itching",
            "erythema",
            "leg swelling",
        ],
        "Endocrine and Metabolic": [
            "dry mouth",
            "excessive thirst",
            "increased appetite",
            "blurred vision",
            "vision loss",
            "palpitations",
            "insomnia",
            "anxiety",
            "depressed mood",
        ],
        "Ear and Hearing": [
            "ear pain",
            "tinnitus",
            "hearing loss",
        ],
        "Musculoskeletal": [
            "joint pain",
            "muscle pain",
        ],
    },
}

MANUAL_CASE_IDS = {item["id"] for item in MANUAL_CASES}

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


def get_symptom_label(symptom: str, lang: str) -> str:
    if lang == "ru":
        return DISPLAY_RU.get(symptom, symptom)
    return symptom


def display_specialty_name(raw_specialty: str, lang: str) -> str:
    if not raw_specialty:
        return ""
    return format_specialty_name(raw_specialty, lang)


def init_state():
    if "graph" not in st.session_state:
        st.session_state.graph = load_graph()
    if "lang" not in st.session_state:
        st.session_state.lang = "ru"
    if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = []
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_input_text" not in st.session_state:
        st.session_state.last_input_text = ""
    if "status_message" not in st.session_state:
        st.session_state.status_message = ""
    if "reset_requested" not in st.session_state:
        st.session_state.reset_requested = False
    if "ml_predictor" not in st.session_state:
        st.session_state.ml_predictor = None
    if "ml_error" not in st.session_state:
        st.session_state.ml_error = ""
    if "ml_specialty_result" not in st.session_state:
        st.session_state.ml_specialty_result = []
    if "ml_case_result" not in st.session_state:
        st.session_state.ml_case_result = []

    if st.session_state.ml_predictor is None and not st.session_state.ml_error:
        try:
            st.session_state.ml_predictor = SpecialtyMLPredictor(st.session_state.graph)
        except Exception as e:
            st.session_state.ml_predictor = None
            st.session_state.ml_error = f"{type(e).__name__}: {e}"


def process_pending_reset():
    if not st.session_state.get("reset_requested", False):
        return

    st.session_state.selected_symptoms = []
    st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    st.session_state.last_result = None
    st.session_state.last_input_text = ""
    st.session_state.status_message = t("reset_done")
    st.session_state.ml_specialty_result = []
    st.session_state.ml_case_result = []

    for key in list(st.session_state.keys()):
        if key.startswith("symptom_"):
            del st.session_state[key]

    st.session_state.reset_requested = False


def request_reset():
    st.session_state.reset_requested = True


def format_state_symptoms(values, lang: str):
    if not values:
        return "—"
    return ", ".join(get_symptom_label(v, lang) for v in values)


def filter_manual_cases(result: dict) -> dict:
    if not result:
        return result
    result = dict(result)
    cases = result.get("cases", [])
    result["cases"] = [case for case in cases if case.get("raw_case") in MANUAL_CASE_IDS]
    return result


def sync_selected_from_checkboxes():
    selected = []
    groups = CURATED_GROUPS[st.session_state.lang]
    for symptoms in groups.values():
        for symptom in symptoms:
            if st.session_state.get(f"symptom_{symptom}", False):
                selected.append(symptom)
    st.session_state.selected_symptoms = selected


def apply_selected_to_checkboxes():
    selected = set(st.session_state.selected_symptoms)
    all_symptoms = set()
    for symptoms in CURATED_GROUPS["ru"].values():
        all_symptoms.update(symptoms)
    for symptom in all_symptoms:
        key = f"symptom_{symptom}"
        if key not in st.session_state:
            st.session_state[key] = symptom in selected


def run_ml_prediction(symptoms):
    predictor = st.session_state.ml_predictor
    if predictor is None or not predictor.is_trained:
        st.session_state.ml_specialty_result = []
        st.session_state.ml_case_result = []
        return

    st.session_state.ml_specialty_result = predictor.predict_top_specialties(symptoms, top_k=3)
    st.session_state.ml_case_result = predictor.predict_top_cases(symptoms, top_k=3)


def run_analysis_from_selection():
    selected = st.session_state.selected_symptoms
    if not selected:
        st.session_state.status_message = t("need_selection")
        return

    st.session_state.patient_data = DEFAULT_PATIENT_DATA.copy()
    input_text = ", ".join(selected)
    st.session_state.last_input_text = ", ".join(
        get_symptom_label(symptom, st.session_state.lang) for symptom in selected
    )

    result = analyze_case(
        input_text,
        st.session_state.graph,
        st.session_state.patient_data,
        lang=st.session_state.lang,
        mutate=True,
    )
    result = filter_manual_cases(result)
    st.session_state.last_result = result
    run_ml_prediction(selected)
    st.session_state.status_message = t("selection_sent")


def answer_pending(answer_text: str):
    result = analyze_case(
        answer_text,
        st.session_state.graph,
        st.session_state.patient_data,
        lang=st.session_state.lang,
        mutate=True,
    )
    result = filter_manual_cases(result)
    st.session_state.last_result = result
    run_ml_prediction(st.session_state.patient_data.get("confirmed", []))


def render_ml_specialty_block():
    st.markdown(f"#### {t('ml_title')}")

    if st.session_state.ml_error:
        st.error(t("ml_error_title"))
        with st.expander(t("ml_debug"), expanded=False):
            st.code(st.session_state.ml_error)
        return

    predictor = st.session_state.ml_predictor
    if predictor is None:
        st.info(t("ml_unavailable"))
        return

    if not predictor.is_trained:
        st.info(t("ml_not_enough"))
        return

    ml_results = st.session_state.ml_specialty_result or []
    if not ml_results:
        st.write("—")
        return

    best = ml_results[0]
    st.markdown(f"**{t('ml_best')}:**")
    st.success(
        f"{display_specialty_name(best['specialty'], st.session_state.lang)} — "
        f"{best['probability']}%"
    )

    st.markdown(f"**{t('ml_top')}:**")
    for item in ml_results:
        title = display_specialty_name(item["specialty"], st.session_state.lang)
        with st.expander(f"{title} — {item['probability']}%"):
            st.progress(min(max(item["probability"] / 100, 0.0), 1.0))
            st.write(f"{t('ml_probability')}: {item['probability']}%")
            st.write(f"{t('ml_cases')}: {item['training_cases']}")


def render_ml_case_block():
    st.markdown(f"#### {t('ml_case_title')}")

    if st.session_state.ml_error:
        st.error(t("ml_error_title"))
        with st.expander(t("ml_debug"), expanded=False):
            st.code(st.session_state.ml_error)
        return

    predictor = st.session_state.ml_predictor
    if predictor is None:
        st.info(t("ml_unavailable"))
        return

    if not predictor.is_trained:
        st.info(t("ml_not_enough"))
        return

    ml_cases = st.session_state.ml_case_result or []
    if not ml_cases:
        st.write("—")
        return

    best = ml_cases[0]
    best_title = best["title_ru"] if st.session_state.lang == "ru" else best["title_en"]
    st.markdown(f"**{t('ml_case_best')}:**")
    st.success(f"{best_title} — {best['probability']}%")

    st.markdown(f"**{t('ml_case_top')}:**")
    for item in ml_cases:
        title = item["title_ru"] if st.session_state.lang == "ru" else item["title_en"]
        specialty = display_specialty_name(item["specialty"], st.session_state.lang)
        header = f"{title} — {item['probability']}%"
        if specialty:
            header += f" ({specialty})"

        with st.expander(header):
            st.progress(min(max(item["probability"] / 100, 0.0), 1.0))
            st.write(f"{t('ml_probability')}: {item['probability']}%")
            st.write(f"{t('ml_cases')}: {item['training_cases']}")
            if item.get("symptoms"):
                st.write(f"{t('ml_features')}:")
                for symptom in item["symptoms"]:
                    st.write(f"- {get_symptom_label(symptom, st.session_state.lang)}")

def render_consensus_block():
    st.markdown("Согласованность системы")

    result = st.session_state.last_result
    ml_results = st.session_state.ml_specialty_result

    if not result or not result.get("specialties") or not ml_results:
        st.write("—")
        return

    # rule-based врач (первый)
    rule_spec = result["specialties"][0]["name"].lower()

    # ML топ-1
    ml_spec = ml_results[0]["specialty"].lower()

    if rule_spec in ml_spec or ml_spec in rule_spec:
        st.success("Совпадение: логика и ML дают одинаковое направление → высокая уверенность")
    else:
        st.warning(
            f"Разные гипотезы:\n\n"
            f"• Логика: {result['specialties'][0]['name']}\n"
            f"• ML: {ml_results[0]['specialty']}\n\n"
            f"ML предлагает альтернативную интерпретацию симптомов"
        )

st.set_page_config(page_title="MedDiognist", layout="wide")
init_state()
process_pending_reset()
apply_selected_to_checkboxes()

st.title(t("app_title"))
st.caption(t("subtitle"))
st.caption(t("disclaimer"))

with st.sidebar:
    st.markdown(f"### {t('language')}")
    previous_lang = st.session_state.lang
    st.session_state.lang = st.radio(
        t("language"),
        ["ru", "en"],
        index=0 if st.session_state.lang == "ru" else 1,
        horizontal=True,
    )

    if previous_lang != st.session_state.lang:
        st.rerun()

    st.markdown(f"### {t('constructor')}")
    st.caption(t("symptom_groups"))

    for group_name, symptoms in CURATED_GROUPS[st.session_state.lang].items():
        with st.expander(group_name, expanded=False):
            for symptom in symptoms:
                st.checkbox(
                    get_symptom_label(symptom, st.session_state.lang),
                    key=f"symptom_{symptom}",
                    on_change=sync_selected_from_checkboxes,
                )

    if st.button(t("analyze"), use_container_width=True):
        sync_selected_from_checkboxes()
        run_analysis_from_selection()
        st.rerun()

    if st.button(t("clear"), use_container_width=True):
        request_reset()
        st.rerun()

    st.markdown(f"### {t('state')}")
    st.write(f"**{t('confirmed')}:** {format_state_symptoms(st.session_state.patient_data['confirmed'], st.session_state.lang)}")
    st.write(f"**{t('denied')}:** {format_state_symptoms(st.session_state.patient_data['denied'], st.session_state.lang)}")

    pending = st.session_state.patient_data["pending_question"]
    pending_display = get_symptom_label(pending, st.session_state.lang) if pending else "—"
    st.write(f"**{t('pending')}:** {pending_display}")

    current_triage = st.session_state.patient_data["triage"]
    triage_display = triage_label(current_triage, st.session_state.lang) if current_triage else "—"
    st.write(f"**{t('triage')}:** {triage_display}")

left, right = st.columns([1, 1])

with left:
    st.subheader(t("constructor"))
    if st.session_state.selected_symptoms:
        st.markdown(f"**{t('selection')}:**")
        for symptom in st.session_state.selected_symptoms:
            st.write(f"- {get_symptom_label(symptom, st.session_state.lang)}")
    else:
        st.info(t("start_hint"))

    if st.session_state.status_message:
        if st.session_state.status_message == t("need_selection"):
            st.warning(st.session_state.status_message)
        else:
            st.success(st.session_state.status_message)

    result = st.session_state.last_result
    if result and result.get("question"):
        st.markdown(f"#### {t('pending_question')}")
        st.write(result["question"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button(t("answer_yes"), use_container_width=True):
                answer_pending("да" if st.session_state.lang == "ru" else "yes")
                st.rerun()
        with c2:
            if st.button(t("answer_no"), use_container_width=True):
                answer_pending("нет" if st.session_state.lang == "ru" else "no")
                st.rerun()

with right:
    st.subheader(t("analysis"))
    result = st.session_state.last_result

    if result and result.get("action") == "emergency":
        st.error(t("danger_alert"))

    if result and result.get("triage"):
        st.metric(t("risk_level"), triage_label(result["triage"], st.session_state.lang))

    if result and result.get("summary"):
        st.markdown(f"#### {t('summary')}")
        st.write(result["summary"])

    render_ml_specialty_block()
    render_ml_case_block()
    render_consensus_block()

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
        st.markdown(f"#### {t('specialties')}")
        for item in result["specialties"]:
            name = item["name"]
            if st.session_state.lang == "ru" and all(ord(ch) < 128 for ch in name if ch.isalpha()):
                name = display_specialty_name(name.lower(), st.session_state.lang) or name
            st.write(f"- {name}")

    if result and result.get("cases"):
        st.markdown(f"#### {t('cases')}")
        for item in result["cases"]:
            title = item["case"]
            if item.get("specialty"):
                title += f" ({item['specialty']})"
            with st.expander(f"{title} — {item['score']}%"):
                st.progress(min(max(item["score"] / 100, 0.0), 1.0))
                st.caption(f"{t('probability')}: {item['score']}%")
                if item.get("matches"):
                    st.write(f"{t('matched')}:")
                    for match in item["matches"]:
                        st.write(f"- {match}")