import re
from difflib import get_close_matches
from collections import defaultdict
from knowledge_graph import SYNONYMS_RU_TO_EN, DISPLAY_RU

YES_WORDS = {"да", "есть", "ага", "угу", "конечно", "yes", "y"}
NO_WORDS = {"нет", "неа", "не", "no", "n"}

DANGER_TERMS = {
    "chest pain",
    "shortness of breath",
    "seizure",
    "loss of consciousness",
    "bleeding",
    "epistaxis",
    "hematuria",
    "blood in stool",
}

SYMPTOM_WEIGHTS = {
    "chest pain": 3.0,
    "shortness of breath": 3.0,
    "seizure": 3.2,
    "loss of consciousness": 3.2,
    "bleeding": 3.0,
    "epistaxis": 2.2,
    "hematuria": 2.8,
    "blood in stool": 2.8,
    "fever": 1.6,
    "chills": 1.4,
    "fatigue": 1.0,
    "weakness": 1.0,
    "headache": 1.1,
    "dizziness": 1.2,
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
    "urinary frequency": 1.7,
    "numbness": 2.0,
    "ear pain": 1.4,
    "tinnitus": 1.2,
    "hearing loss": 1.8,
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def apply_ru_to_en(text: str) -> str:
    for ru, en in SYNONYMS_RU_TO_EN.items():
        if ru in text:
            text = text.replace(ru, en)
    return text

def extract_symptoms(text: str, graph, fuzzy_cutoff: float = 0.84):
    text = normalize_text(text)
    text = apply_ru_to_en(text)

    symptom_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "symptom"]
    found = set()

    for s in symptom_nodes:
        if s in text:
            found.add(s)

    tokens = re.findall(r"[a-zа-яё]+", text)
    for tok in tokens:
        if tok in symptom_nodes:
            found.add(tok)
            continue
        close = get_close_matches(tok, symptom_nodes, n=1, cutoff=fuzzy_cutoff)
        if close:
            found.add(close[0])

    return sorted(found)

def symptom_weight(symptom: str) -> float:
    return SYMPTOM_WEIGHTS.get(symptom, 1.0)

def calculate_triage(confirmed_symptoms):
    confirmed = set(confirmed_symptoms)

    if confirmed & {"chest pain", "shortness of breath", "seizure", "loss of consciousness", "bleeding"}:
        return "CRITICAL"

    if len(confirmed & {"hematuria", "blood in stool", "epistaxis"}) >= 1:
        return "HIGH"

    if len(confirmed & {"fever", "cough", "shortness of breath"}) >= 2:
        return "HIGH"

    if len(confirmed & {"vomiting", "diarrhea", "abdominal pain"}) >= 2:
        return "MEDIUM"

    if len(confirmed & {"headache", "fatigue", "sore throat", "rhinorrhea", "sneezing"}) >= 2:
        return "LOW"

    score = sum(symptom_weight(s) for s in confirmed)

    if score >= 6:
        return "HIGH"
    if score >= 3.5:
        return "MEDIUM"
    return "LOW"

def triage_label(triage: str, lang: str = "ru") -> str:
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
        advice = {
            "LOW": "Monitor symptoms and rest.",
            "MEDIUM": "Consider scheduling a doctor consultation.",
            "HIGH": "Medical consultation is recommended as soon as possible.",
            "CRITICAL": "Seek urgent medical care immediately.",
        }
        return advice[triage]

    advice = {
        "LOW": "Наблюдайте за симптомами и соблюдайте покой.",
        "MEDIUM": "Рекомендуется консультация врача.",
        "HIGH": "Желательно обратиться к врачу как можно скорее.",
        "CRITICAL": "Нужно срочно обратиться за неотложной медицинской помощью.",
    }
    return advice[triage]

def score_cases(graph, confirmed_symptoms):
    confirmed = set(confirmed_symptoms)
    results = []

    for node, data in graph.nodes(data=True):
        if data.get("type") != "case":
            continue

        neighbors = list(graph.neighbors(node))
        kws = [n for n in neighbors if graph.nodes[n].get("type") == "symptom"]
        if not kws:
            continue

        matches = sorted(list(confirmed & set(kws)))
        if not matches:
            continue

        matched_weight = sum(symptom_weight(m) for m in matches)
        total_weight = sum(symptom_weight(k) for k in kws)
        weighted_score = matched_weight / total_weight if total_weight else 0

        idf_bonus = 0.0
        for m in matches:
            idf_bonus += float(graph.nodes[m].get("idf", 1.0))
        idf_bonus = idf_bonus / max(len(matches), 1)

        final_score = weighted_score * 0.8 + min(idf_bonus / 5.0, 1.0) * 0.2

        results.append((node, final_score, matches, kws))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def best_next_question(graph, top_case, confirmed, denied):
    neighbors = list(graph.neighbors(top_case))
    kws = [n for n in neighbors if graph.nodes[n].get("type") == "symptom"]
    missing = [k for k in kws if k not in confirmed and k not in denied]
    if not missing:
        return None

    def score_question(k):
        idf = float(graph.nodes[k].get("idf", 1.0))
        w = symptom_weight(k)
        generic_penalty = 0.7 if k in {"fever", "cough", "fatigue", "headache"} else 1.0
        return idf * w * generic_penalty

    missing.sort(key=score_question, reverse=True)
    return missing[0]

def format_symptom(symptom: str, lang: str, graph) -> str:
    if lang == "ru":
        ru = graph.nodes[symptom].get("ru") or DISPLAY_RU.get(symptom) or symptom
        return ru
    return symptom

def format_specialty(spec: str, lang: str) -> str:
    if lang == "en":
        return spec

    mapping = {
        "pulmonary": "Пульмонолог",
        "cardiology": "Кардиолог",
        "neurology": "Невролог",
        "gastroenterology": "Гастроэнтеролог",
        "urology": "Уролог",
        "otolaryngology": "ЛОР",
        "orthopedic": "Ортопед",
        "allergy / immunology": "Аллерголог / иммунолог",
        "internal medicine": "Терапевт",
        "emergency room reports": "Неотложная помощь",
    }
    return mapping.get(spec, spec)

def process_text_message(text: str, graph, patient_data: dict, lang: str = "ru") -> str:
    text = normalize_text(text)

    patient_data.setdefault("confirmed", [])
    patient_data.setdefault("denied", [])
    patient_data.setdefault("pending_question", None)
    patient_data.setdefault("emergency_triggered", False)
    patient_data.setdefault("emergency_term", None)
    patient_data.setdefault("triage", None)

    if text == "/reset":
        return "__RESET__"

    if patient_data["emergency_triggered"]:
        if lang == "ru":
            return (
                f"🚨 Ранее найден опасный признак: **{format_symptom(patient_data['emergency_term'], lang, graph)}**.\n\n"
                "Я остановлю подсказки. Лучше обратиться к врачу.\n\n"
                "Новый случай: `/reset`."
            )
        return (
            f"🚨 Previously detected a dangerous sign: **{patient_data['emergency_term']}**.\n\n"
            "I will stop. Please seek medical care.\n\n"
            "New case: `/reset`."
        )

    if text in YES_WORDS and patient_data["pending_question"]:
        s = patient_data["pending_question"]
        if s not in patient_data["confirmed"]:
            patient_data["confirmed"].append(s)
        patient_data["pending_question"] = None

    if text in NO_WORDS and patient_data["pending_question"]:
        s = patient_data["pending_question"]
        if s not in patient_data["denied"]:
            patient_data["denied"].append(s)
        patient_data["pending_question"] = None

    extracted = extract_symptoms(text, graph)
    for s in extracted:
        if s not in patient_data["confirmed"]:
            patient_data["confirmed"].append(s)

    for s in patient_data["confirmed"]:
        if s in DANGER_TERMS:
            patient_data["emergency_triggered"] = True
            patient_data["emergency_term"] = s
            patient_data["triage"] = "CRITICAL"

            if lang == "ru":
                return (
                    f"🚨 Потенциально опасный симптом: **{format_symptom(s, lang, graph)}**.\n\n"
                    "Рекомендуется **срочно** обратиться к врачу/в неотложку.\n\n"
                    "Новый случай: `/reset`."
                )
            return (
                f"🚨 Potentially dangerous symptom: **{s}**.\n\n"
                "Please seek urgent medical care.\n\n"
                "New case: `/reset`."
            )

    if not patient_data["confirmed"]:
        return (
            "Опиши симптомы (RU/EN). Пример: `кашель температура`."
            if lang == "ru"
            else "Describe symptoms (RU/EN). Example: `cough fever`."
        )

    triage = calculate_triage(patient_data["confirmed"])
    patient_data["triage"] = triage

    scored = score_cases(graph, patient_data["confirmed"])
    if not scored:
        if lang == "ru":
            return (
                f"🚦 Уровень риска: **{triage_label(triage, lang)}**.\n"
                f"Совет: {triage_advice(triage, lang)}\n\n"
                "Не нахожу похожих кейсов. Добавь ещё 2–3 симптома."
            )
        return (
            f"🚦 Risk level: **{triage_label(triage, lang)}**.\n"
            f"Advice: {triage_advice(triage, lang)}\n\n"
            "No similar cases found yet. Add 2–3 more symptoms."
        )

    top3 = scored[:3]

    spec_scores = defaultdict(float)
    for case, score, _, _ in top3:
        spec = graph.nodes[case].get("specialty", "")
        if spec:
            spec_scores[spec] += score
    best_specs = sorted(spec_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    if lang == "ru":
        lines = [
            f"🚦 **Уровень риска:** {triage_label(triage, lang)}",
            f"💡 **Рекомендация:** {triage_advice(triage, lang)}",
            "",
            "🩺 **Похожие медицинские кейсы (top-3):**"
        ]
    else:
        lines = [
            f"🚦 **Risk level:** {triage_label(triage, lang)}",
            f"💡 **Advice:** {triage_advice(triage, lang)}",
            "",
            "🩺 **Similar medical cases (top-3):**"
        ]

    for case, score, matches, _all in top3:
        percent = round(score * 100)
        spec = graph.nodes[case].get("specialty", "")
        spec_txt = f" ({format_specialty(spec, lang)})" if spec else ""

        if lang == "ru":
            lines.append(f"- **{case}**{spec_txt} — {percent}%")
            lines.append("  ✅ Совпало: " + ", ".join(format_symptom(m, lang, graph) for m in matches))
        else:
            lines.append(f"- **{case}**{spec_txt} — {percent}%")
            lines.append("  ✅ Matched: " + ", ".join(matches))

    if best_specs:
        if lang == "ru":
            lines.append("")
            lines.append("🧭 **К какому специалисту обратиться:**")
            for s, v in best_specs:
                lines.append(f"- {format_specialty(s, lang)} ({v:.2f})")
        else:
            lines.append("")
            lines.append("🧭 **Suggested specialty:**")
            for s, v in best_specs:
                lines.append(f"- {format_specialty(s, lang)} ({v:.2f})")

    top_case = top3[0][0]
    q = best_next_question(graph, top_case, set(patient_data["confirmed"]), set(patient_data["denied"]))
    if q:
        patient_data["pending_question"] = q
        if lang == "ru":
            lines.append("")
            lines.append(f"Уточнение: **есть ли у вас {format_symptom(q, lang, graph)}?** (да/нет)")
        else:
            lines.append("")
            lines.append(f"Question: **do you have {q}?** (yes/no)")

    lines.append("")
    lines.append(
        "⚠️ Это учебный помощник, не медицинский диагноз."
        if lang == "ru"
        else "⚠️ Educational assistant, not a medical diagnosis."
    )

    return "\n".join(lines)