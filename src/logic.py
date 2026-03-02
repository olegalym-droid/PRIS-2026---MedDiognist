import re
from difflib import get_close_matches
from knowledge_graph import SYNONYMS

DANGER_SYMPTOMS = {
    "кровотечение",
    "носовое кровотечение",
    "кровь в стуле",
    "кровь в моче",
    "потеря сознания",
    "судороги",
    "боль в груди",
    "одышка",
}

YES_WORDS = {"да", "есть", "ага", "угу", "конечно", "давай"}
NO_WORDS = {"нет", "неа", "не", "не было", "не наблюдаю"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _apply_synonyms(text: str) -> str:
    for phrase, canonical in SYNONYMS.items():
        if phrase in text:
            text = text.replace(phrase, canonical)
    return text


def extract_symptoms(text: str, graph, fuzzy_cutoff: float = 0.84):
    text = normalize_text(text)
    text = _apply_synonyms(text)

    symptom_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "symptom"]
    found = set()

    for s in symptom_nodes:
        if s in text:
            found.add(s)

    tokens = re.findall(r"[а-яё]+", text)
    for tok in tokens:
        if tok in symptom_nodes:
            found.add(tok)
            continue

        close = get_close_matches(tok, symptom_nodes, n=1, cutoff=fuzzy_cutoff)
        if close:
            found.add(close[0])

    return sorted(found)


def score_diseases(graph, confirmed_symptoms):
    results = []
    confirmed = set(confirmed_symptoms)

    for node, data in graph.nodes(data=True):
        if data.get("type") != "disease":
            continue

        disease_symptoms = [n for n in graph.neighbors(node) if graph.nodes[n].get("type") == "symptom"]
        if not disease_symptoms:
            continue

        matches = sorted(list(confirmed & set(disease_symptoms)))
        if matches:
            score = len(matches) / len(disease_symptoms)
            results.append((node, score, matches, disease_symptoms))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_medicines_for_disease(graph, disease):
    meds = []
    for n in graph.neighbors(disease):
        if graph.nodes[n].get("type") == "medicine":
            meds.append(n)
    return sorted(meds)


def process_text_message(text: str, graph, patient_data: dict) -> str:
    text = normalize_text(text)

    patient_data.setdefault("confirmed", [])
    patient_data.setdefault("denied", [])
    patient_data.setdefault("pending_question", None)
    patient_data.setdefault("emergency_triggered", False)
    patient_data.setdefault("emergency_symptom", None)

    if text == "/reset":
        return "__RESET__"

    if patient_data["emergency_triggered"]:
        return (
            f"🚨 Ранее обнаружен опасный симптом: **{patient_data['emergency_symptom']}**.\n\n"
            "Я не продолжу диагностику, чтобы не вводить в заблуждение.\n\n"
            "Лучшее действие — обратиться к врачу.\n\n"
            "Новый случай: `/reset`."
        )

    if text in YES_WORDS and patient_data["pending_question"]:
        sym = patient_data["pending_question"]
        if sym not in patient_data["confirmed"]:
            patient_data["confirmed"].append(sym)
        patient_data["pending_question"] = None

    if text in NO_WORDS and patient_data["pending_question"]:
        sym = patient_data["pending_question"]
        if sym not in patient_data["denied"]:
            patient_data["denied"].append(sym)
        patient_data["pending_question"] = None

    extracted = extract_symptoms(text, graph)
    for s in extracted:
        if s not in patient_data["confirmed"]:
            patient_data["confirmed"].append(s)

    for s in patient_data["confirmed"]:
        if s in DANGER_SYMPTOMS:
            patient_data["emergency_triggered"] = True
            patient_data["emergency_symptom"] = s
            return (
                f"🚨 Обнаружен потенциально опасный симптом: **{s}**.\n\n"
                "Рекомендуется **срочно** обратиться к врачу/в неотложку.\n\n"
                "Новый случай: `/reset`."
            )

    if not patient_data["confirmed"]:
        return "Опиши симптомы (пример: `кашель, температура, боль в горле`)."

    scored = score_diseases(graph, patient_data["confirmed"])
    if not scored:
        return "Пока не могу сопоставить заболевание. Добавь ещё симптомы (2–3 штуки)."

    top3 = scored[:3]
    top_disease, top_score, top_matches, top_all = top3[0]

    lines = ["🩺 **Топ вероятных вариантов:**"]
    for disease, score, matches, all_symptoms in top3:
        percent = round(score * 100)
        lines.append(f"- **{disease}** — {percent}%")
        lines.append(f"  ✅ Совпало: {', '.join(matches)}")

        missing_for_this = [s for s in all_symptoms if s not in patient_data["confirmed"] and s not in patient_data["denied"]]
        if missing_for_this:
            lines.append(f"  ❓ Не хватает: {', '.join(missing_for_this[:3])}")

        meds = get_medicines_for_disease(graph, disease)
        if meds:
            lines.append(f"  💊 Лекарства (примерно): {', '.join(meds[:3])}")

    missing_top = [s for s in top_all if s not in patient_data["confirmed"] and s not in patient_data["denied"]]
    if missing_top:
        q = missing_top[0]
        patient_data["pending_question"] = q
        lines.append("")
        lines.append(f"Уточнение: **есть ли у вас {q}?** (да/нет)")
    else:
        meds = get_medicines_for_disease(graph, top_disease)
        if meds:
            lines.append("")
            lines.append(f"💊 Возможные лекарства для **{top_disease}**: {', '.join(meds)}")

    lines.append("")
    lines.append("⚠️ Это учебный помощник, не медицинский диагноз.")

    return "\n".join(lines)