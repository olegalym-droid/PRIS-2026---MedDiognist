# src/logic.py

import copy
import re
from collections import defaultdict
from difflib import get_close_matches

from knowledge_graph import (
    DISPLAY_RU,
    SYNONYMS_RU_TO_EN,
    format_case_title,
    format_specialty_name,
)

YES_WORDS = {"да", "есть", "ага", "угу", "конечно", "yes", "y", "yeah", "yep"}
NO_WORDS = {"нет", "неа", "не", "no", "n", "nope"}

DANGER_TERMS = {
    "chest pain",
    "shortness of breath",
    "seizure",
    "loss of consciousness",
    "bleeding",
    "epistaxis",
    "hematuria",
    "blood in stool",
    "confusion",
    "jaundice",
    "vision loss",
}

SYMPTOM_WEIGHTS = {
    "chest pain": 3.0,
    "shortness of breath": 3.0,
    "seizure": 3.2,
    "loss of consciousness": 3.2,
    "bleeding": 3.0,
    "epistaxis": 2.4,
    "hematuria": 2.8,
    "blood in stool": 2.8,
    "fever": 1.6,
    "chills": 1.4,
    "fatigue": 1.0,
    "weakness": 1.1,
    "headache": 1.1,
    "dizziness": 1.3,
    "cough": 1.2,
    "dry cough": 1.3,
    "productive cough": 1.5,
    "rhinorrhea": 0.9,
    "nasal congestion": 0.9,
    "sneezing": 0.8,
    "sore throat": 1.1,
    "nausea": 1.2,
    "vomiting": 1.6,
    "diarrhea": 1.7,
    "constipation": 1.0,
    "abdominal pain": 1.8,
    "heartburn": 1.0,
    "bloating": 0.9,
    "back pain": 1.2,
    "flank pain": 1.8,
    "joint pain": 1.3,
    "muscle pain": 1.1,
    "rash": 1.4,
    "itching": 1.0,
    "erythema": 1.1,
    "dysuria": 2.0,
    "urinary frequency": 1.8,
    "numbness": 2.0,
    "ear pain": 1.4,
    "tinnitus": 1.2,
    "hearing loss": 1.8,
    "memory loss": 2.2,
    "forgetfulness": 1.9,
    "confusion": 2.8,
    "drowsiness": 1.6,
    "dry mouth": 1.2,
    "excessive thirst": 2.0,
    "increased appetite": 1.3,
    "blurred vision": 2.0,
    "vision loss": 3.0,
    "jaundice": 3.0,
    "edema": 1.7,
    "leg swelling": 1.9,
    "weight loss": 1.8,
    "weight gain": 1.5,
    "tremor": 1.7,
    "palpitations": 2.0,
    "insomnia": 1.1,
    "anxiety": 1.1,
    "depressed mood": 1.0,
}

SPECIALTY_HINTS = {
    "endocrinology": {"excessive thirst", "urinary frequency", "weight loss", "blurred vision", "weight gain", "tremor", "palpitations"},
    "neurology": {"memory loss", "forgetfulness", "confusion", "drowsiness", "headache", "dizziness", "numbness", "seizure"},
    "hepatology": {"jaundice", "abdominal pain", "nausea", "fatigue", "itching"},
    "ophthalmology": {"blurred vision", "vision loss", "headache"},
    "cardiology": {"chest pain", "shortness of breath", "palpitations", "dizziness", "leg swelling"},
    "gastroenterology": {"abdominal pain", "nausea", "vomiting", "diarrhea", "blood in stool", "heartburn", "bloating"},
    "urology": {"dysuria", "urinary frequency", "hematuria", "flank pain"},
    "pulmonary": {"cough", "dry cough", "productive cough", "shortness of breath", "fever"},
    "allergy / immunology": {"rash", "itching", "sneezing", "rhinorrhea", "nasal congestion", "erythema"},
    "internal medicine": {"fever", "fatigue", "weakness", "headache", "dizziness", "cough", "nausea"},
}

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[,_;/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def apply_ru_to_en(text: str) -> str:
    normalized = text
    for ru, en in sorted(SYNONYMS_RU_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(ru)}\b", en, normalized)
    return normalized


def get_symptom_nodes(graph):
    return [node for node, data in graph.nodes(data=True) if data.get("type") == "symptom"]


def extract_symptoms(text: str, graph, fuzzy_cutoff: float = 0.88):
    text = normalize_text(text)
    text = apply_ru_to_en(text)

    symptom_nodes = get_symptom_nodes(graph)
    found = set()

    padded_text = f" {text} "
    for symptom in sorted(symptom_nodes, key=len, reverse=True):
        if f" {symptom} " in padded_text:
            found.add(symptom)

    tokens = re.findall(r"[a-zа-яё]+", text)
    for token in tokens:
        if token in symptom_nodes:
            found.add(token)
            continue
        close = get_close_matches(token, symptom_nodes, n=1, cutoff=fuzzy_cutoff)
        if close:
            found.add(close[0])

    return sorted(found)


def symptom_weight(symptom: str) -> float:
    return SYMPTOM_WEIGHTS.get(symptom, 1.0)


def calculate_triage(confirmed_symptoms):
    confirmed = set(confirmed_symptoms)

    if confirmed & {"chest pain", "shortness of breath", "seizure", "loss of consciousness", "bleeding"}:
        return "CRITICAL"

    if confirmed & {"hematuria", "blood in stool", "epistaxis", "confusion", "jaundice", "vision loss"}:
        return "HIGH"

    if len(confirmed & {"fever", "cough", "shortness of breath"}) >= 2:
        return "HIGH"

    if len(confirmed & {"vomiting", "diarrhea", "abdominal pain"}) >= 2:
        return "MEDIUM"

    if len(confirmed & {"memory loss", "forgetfulness", "confusion"}) >= 2:
        return "HIGH"

    if len(confirmed & {"excessive thirst", "urinary frequency", "weight loss", "blurred vision"}) >= 2:
        return "MEDIUM"

    score = sum(symptom_weight(symptom) for symptom in confirmed)
    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def triage_label(triage: str, lang: str = "ru") -> str:
    if not triage:
        return ""
    if lang == "en":
        return triage
    mapping = {
        "LOW": "Низкий",
        "MEDIUM": "Средний",
        "HIGH": "Высокий",
        "CRITICAL": "Критический",
    }
    return mapping.get(triage, triage)


def triage_advice(triage: str, lang: str = "ru") -> str:
    if lang == "en":
        mapping = {
            "LOW": "Monitor symptoms and rest.",
            "MEDIUM": "Schedule a doctor visit in the near future.",
            "HIGH": "See a doctor as soon as possible.",
            "CRITICAL": "Seek urgent medical care immediately.",
        }
        return mapping.get(triage, "")
    mapping = {
        "LOW": "Наблюдайте за симптомами и соблюдайте покой.",
        "MEDIUM": "Запишитесь на плановый прием к врачу в ближайшее время.",
        "HIGH": "Обратитесь к врачу как можно скорее.",
        "CRITICAL": "Нужно срочно обратиться за неотложной медицинской помощью.",
    }
    return mapping.get(triage, "")


def format_symptom(symptom: str, lang: str, graph) -> str:
    if lang == "ru":
        return graph.nodes[symptom].get("ru") or DISPLAY_RU.get(symptom) or symptom
    return symptom


def format_specialty(spec: str, lang: str) -> str:
    return format_specialty_name(spec, lang)


def format_case(case: str, graph, lang: str) -> str:
    node = graph.nodes.get(case, {})
    if lang == "ru":
        return node.get("title_ru") or format_case_title(case, "ru")
    return node.get("title_en") or format_case_title(case, "en")


def score_cases(graph, confirmed_symptoms):
    confirmed = set(confirmed_symptoms)
    results = []

    for node, data in graph.nodes(data=True):
        if data.get("type") != "case":
            continue

        neighbors = list(graph.neighbors(node))
        case_symptoms = [n for n in neighbors if graph.nodes[n].get("type") == "symptom"]
        if not case_symptoms:
            continue

        matches = sorted(list(confirmed & set(case_symptoms)))
        if not matches:
            continue

        matched_weight = sum(symptom_weight(symptom) for symptom in matches)
        total_weight = sum(symptom_weight(symptom) for symptom in case_symptoms)
        weighted_score = matched_weight / total_weight if total_weight else 0.0

        match_ratio = len(matches) / max(len(case_symptoms), 1)
        idf_bonus = sum(float(graph.nodes[m].get("idf", 1.0)) for m in matches) / max(len(matches), 1)
        source_bonus = 0.08 if data.get("source") == "manual" else 0.0

        final_score = weighted_score * 0.55 + match_ratio * 0.25 + min(idf_bonus / 5.0, 1.0) * 0.20 + source_bonus
        results.append((node, final_score, matches, case_symptoms))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def best_next_question(graph, top_case, confirmed, denied):
    neighbors = list(graph.neighbors(top_case))
    case_symptoms = [n for n in neighbors if graph.nodes[n].get("type") == "symptom"]
    missing = [symptom for symptom in case_symptoms if symptom not in confirmed and symptom not in denied]

    if not missing:
        return None

    def question_score(symptom):
        idf = float(graph.nodes[symptom].get("idf", 1.0))
        weight = symptom_weight(symptom)
        generic_penalty = 0.7 if symptom in {"fever", "cough", "fatigue", "headache"} else 1.0
        return idf * weight * generic_penalty

    missing.sort(key=question_score, reverse=True)
    return missing[0]


def _ensure_patient_data(patient_data: dict):
    patient_data.setdefault("confirmed", [])
    patient_data.setdefault("denied", [])
    patient_data.setdefault("pending_question", None)
    patient_data.setdefault("emergency_triggered", False)
    patient_data.setdefault("emergency_term", None)
    patient_data.setdefault("triage", None)
    return patient_data


def _append_unique(items, value):
    if value not in items:
        items.append(value)


def _rank_specialties(confirmed_symptoms):
    scores = defaultdict(float)
    confirmed = set(confirmed_symptoms)

    for specialty, hints in SPECIALTY_HINTS.items():
        overlap = confirmed & hints
        if overlap:
            scores[specialty] += sum(symptom_weight(symptom) for symptom in overlap) / max(len(hints), 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def generate_recommendation(triage: str, specialties: list, lang: str = "ru") -> str:
    doctor = specialties[0]["name"] if specialties else ("терапевту" if lang == "ru" else "a physician")

    if lang == "en":
        if triage == "CRITICAL":
            return "Immediate action: call emergency services or go to the emergency department now."
        if triage == "HIGH":
            return f"Recommended action: see {doctor} as soon as possible, preferably today."
        if triage == "MEDIUM":
            return f"Recommended action: book an appointment with {doctor} within the next few days."
        return f"Recommended action: start with {doctor} if symptoms persist or worsen."

    if triage == "CRITICAL":
        return "Конкретное действие: срочно вызывайте скорую помощь или немедленно обращайтесь в отделение неотложной помощи."
    if triage == "HIGH":
        return f"Конкретное действие: как можно скорее обратитесь к врачу. Приоритетный специалист: {doctor}."
    if triage == "MEDIUM":
        return f"Конкретное действие: запишитесь на прием к врачу в ближайшие дни. Рекомендуемый специалист: {doctor}."
    return f"Конкретное действие: начните с консультации у врача, если симптомы сохраняются. Оптимально обратиться к специалисту: {doctor}."


def analyze_case(text: str, graph, patient_data: dict, lang: str = "ru", mutate: bool = True) -> dict:
    state = patient_data if mutate else copy.deepcopy(patient_data)
    _ensure_patient_data(state)

    text = normalize_text(text)

    if text == "/reset":
        return {"action": "reset"}

    if state["emergency_triggered"]:
        term = state["emergency_term"]
        return {
            "action": "emergency_locked",
            "triage": "CRITICAL",
            "recognized_symptoms": [format_symptom(s, lang, graph) for s in state["confirmed"]],
            "message": (
                f"Ранее найден опасный признак: {format_symptom(term, lang, graph)}. Диагностика остановлена."
                if lang == "ru"
                else f"A dangerous symptom was already detected: {term}. Diagnostic flow stopped."
            ),
        }

    extracted = extract_symptoms(text, graph)

    if text in YES_WORDS and state["pending_question"]:
        _append_unique(state["confirmed"], state["pending_question"])
        state["pending_question"] = None
    elif text in NO_WORDS and state["pending_question"]:
        _append_unique(state["denied"], state["pending_question"])
        state["pending_question"] = None

    for symptom in extracted:
        _append_unique(state["confirmed"], symptom)

    for symptom in state["confirmed"]:
        if symptom in DANGER_TERMS:
            state["emergency_triggered"] = True
            state["emergency_term"] = symptom
            state["triage"] = "CRITICAL"

            emergency_specialties = [{"name": "Неотложная помощь"}] if lang == "ru" else [{"name": "Emergency care"}]
            return {
                "action": "emergency",
                "triage": "CRITICAL",
                "recognized_symptoms": [format_symptom(s, lang, graph) for s in state["confirmed"]],
                "emergency_term": symptom,
                "advice": triage_advice("CRITICAL", lang),
                "final_recommendation": generate_recommendation("CRITICAL", emergency_specialties, lang),
            }

    if not state["confirmed"]:
        return {
            "action": "need_input",
            "triage": None,
            "recognized_symptoms": [],
            "message": "Опишите симптомы свободным текстом." if lang == "ru" else "Describe symptoms in free text.",
        }

    triage = calculate_triage(state["confirmed"])
    state["triage"] = triage

    scored = score_cases(graph, state["confirmed"])
    specialty_rank = _rank_specialties(state["confirmed"])

    top_cases = []
    for case, score, matches, _ in scored[:3]:
        raw_specialty = graph.nodes[case].get("specialty", "")
        top_cases.append(
            {
                "case": format_case(case, graph, lang),
                "raw_case": case,
                "score": min(round(score * 100), 99),
                "specialty": format_specialty(raw_specialty, lang) if raw_specialty else "",
                "matches": [format_symptom(m, lang, graph) for m in matches],
            }
        )

    if not specialty_rank and top_cases:
        spec_scores = defaultdict(float)
        for case, score, _, _ in scored[:3]:
            raw_specialty = graph.nodes[case].get("specialty", "")
            if raw_specialty:
                spec_scores[raw_specialty] += score
        specialty_rank = sorted(spec_scores.items(), key=lambda item: item[1], reverse=True)

    formatted_specialties = [
        {"name": format_specialty(spec, lang), "score": round(score, 2)}
        for spec, score in specialty_rank[:3]
    ]

    top_case_id = scored[0][0] if scored else None
    next_question = None
    if top_case_id:
        next_question = best_next_question(
            graph,
            top_case_id,
            set(state["confirmed"]),
            set(state["denied"]),
        )
    state["pending_question"] = next_question

    summary_ru = None
    summary_en = None

    if formatted_specialties:
        summary_ru = f"Предварительный вывод: наиболее подходящее направление консультации — {formatted_specialties[0]['name']}."
        summary_en = f"Preliminary conclusion: the most suitable consultation direction is {formatted_specialties[0]['name']}."
    elif top_cases:
        summary_ru = f"Предварительный вывод: наиболее вероятный клинический сценарий — {top_cases[0]['case']}."
        summary_en = f"Preliminary conclusion: the most likely clinical scenario is {top_cases[0]['case']}."

    final_recommendation = generate_recommendation(triage, formatted_specialties, lang)

    return {
        "action": "ok" if top_cases or formatted_specialties else "no_cases",
        "triage": triage,
        "recognized_symptoms": [format_symptom(s, lang, graph) for s in state["confirmed"]],
        "advice": triage_advice(triage, lang),
        "cases": top_cases,
        "specialties": formatted_specialties,
        "question": format_symptom(next_question, lang, graph) if next_question else None,
        "summary": summary_ru if lang == "ru" else summary_en,
        "final_recommendation": final_recommendation,
    }


def result_to_text(result: dict, graph, lang: str = "ru") -> str:
    action = result.get("action")

    if action == "reset":
        return "__RESET__"

    if action in {"need_input", "emergency_locked"}:
        return result.get("message", "")

    if action == "emergency":
        if lang == "ru":
            return "\n".join(
                [
                    f"Уровень риска: {triage_label(result['triage'], lang)}",
                    f"Опасный симптом: {format_symptom(result['emergency_term'], lang, graph)}",
                    f"Рекомендация: {result['advice']}",
                    result["final_recommendation"],
                ]
            )
        return "\n".join(
            [
                f"Risk level: {triage_label(result['triage'], lang)}",
                f"Dangerous symptom: {result['emergency_term']}",
                f"Advice: {result['advice']}",
                result["final_recommendation"],
            ]
        )

    lines = []

    if result.get("triage"):
        if lang == "ru":
            lines.append(f"Уровень риска: {triage_label(result['triage'], lang)}")
            lines.append(f"Общая рекомендация: {result['advice']}")
        else:
            lines.append(f"Risk level: {triage_label(result['triage'], lang)}")
            lines.append(f"General advice: {result['advice']}")

    if result.get("summary"):
        lines.append(result["summary"])

    if result.get("final_recommendation"):
        lines.append(result["final_recommendation"])

    if result.get("recognized_symptoms"):
        prefix = "Распознанные симптомы: " if lang == "ru" else "Recognized symptoms: "
        lines.append(prefix + ", ".join(result["recognized_symptoms"]))

    if result.get("specialties"):
        lines.append("Рекомендуемый врач:" if lang == "ru" else "Recommended doctor:")
        for item in result["specialties"]:
            lines.append(f"- {item['name']}")

    if result.get("cases"):
        lines.append("Подходящие клинические сценарии:" if lang == "ru" else "Relevant clinical scenarios:")
        for item in result["cases"]:
            specialty = f" ({item['specialty']})" if item.get("specialty") else ""
            lines.append(f"- {item['case']}{specialty} — {item['score']}%")
            if item.get("matches"):
                match_prefix = "  Совпадения: " if lang == "ru" else "  Matches: "
                lines.append(match_prefix + ", ".join(item["matches"]))

    if result.get("question"):
        if lang == "ru":
            lines.append(f"Уточняющий вопрос: есть ли у вас {result['question']}?")
        else:
            lines.append(f"Clarifying question: do you have {result['question']}?")

    return "\n".join(lines)


def process_text_message(text: str, graph, patient_data: dict, lang: str = "ru") -> str:
    result = analyze_case(text, graph, patient_data, lang=lang, mutate=True)
    return result_to_text(result, graph, lang)