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

    # прямые вхождения фраз
    for s in symptom_nodes:
        if s in text:
            found.add(s)

    # fuzzy по токенам (для опечаток)
    tokens = re.findall(r"[a-zа-яё]+", text)
    for tok in tokens:
        if tok in symptom_nodes:
            found.add(tok)
            continue
        close = get_close_matches(tok, symptom_nodes, n=1, cutoff=fuzzy_cutoff)
        if close:
            found.add(close[0])

    return sorted(found)

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
        if matches:
            score = len(matches) / len(kws)
            results.append((node, score, matches, kws))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

def best_next_question(graph, top_case, confirmed, denied):
    neighbors = list(graph.neighbors(top_case))
    kws = [n for n in neighbors if graph.nodes[n].get("type") == "symptom"]
    missing = [k for k in kws if k not in confirmed and k not in denied]
    if not missing:
        return None

    # выбираем самый информативный: максимальный idf (реже встречается -> лучше различает)
    def idf(k):
        return float(graph.nodes[k].get("idf", 1.0))

    missing.sort(key=idf, reverse=True)
    return missing[0]

def format_symptom(symptom: str, lang: str, graph) -> str:
    if lang == "ru":
        ru = graph.nodes[symptom].get("ru") or DISPLAY_RU.get(symptom) or symptom
        return ru
    return symptom

def process_text_message(text: str, graph, patient_data: dict, lang: str = "ru") -> str:
    text = normalize_text(text)

    patient_data.setdefault("confirmed", [])
    patient_data.setdefault("denied", [])
    patient_data.setdefault("pending_question", None)
    patient_data.setdefault("emergency_triggered", False)
    patient_data.setdefault("emergency_term", None)

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

    # ответы да/нет на вопрос
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

    # emergency once
    for s in patient_data["confirmed"]:
        if s in DANGER_TERMS:
            patient_data["emergency_triggered"] = True
            patient_data["emergency_term"] = s
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
        return ("Опиши симптомы (RU/EN). Пример: `кашель температура`."
                if lang == "ru"
                else "Describe symptoms (RU/EN). Example: `cough fever`.")

    scored = score_cases(graph, patient_data["confirmed"])
    if not scored:
        return ("Не нахожу похожих кейсов. Добавь ещё 2–3 симптома."
                if lang == "ru"
                else "No similar cases found yet. Add 2–3 more symptoms.")

    top3 = scored[:3]

    # классификация по specialty (взвешенный подсчёт)
    spec_scores = defaultdict(float)
    for case, score, _, _ in top3:
        spec = graph.nodes[case].get("specialty", "")
        if spec:
            spec_scores[spec] += score
    best_specs = sorted(spec_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    if lang == "ru":
        lines = ["🩺 **Похожие медицинские кейсы (top-3):**"]
    else:
        lines = ["🩺 **Similar medical cases (top-3):**"]

    for case, score, matches, _all in top3:
        percent = round(score * 100)
        spec = graph.nodes[case].get("specialty", "")
        spec_txt = f" ({spec})" if spec else ""
        if lang == "ru":
            lines.append(f"- **{case}**{spec_txt} — {percent}%")
            lines.append("  ✅ Совпало: " + ", ".join(format_symptom(m, lang, graph) for m in matches))
        else:
            lines.append(f"- **{case}**{spec_txt} — {percent}%")
            lines.append("  ✅ Matched: " + ", ".join(matches))

    if best_specs:
        if lang == "ru":
            lines.append("")
            lines.append("🧭 **К какому специалисту (оценка по top-кейсам):**")
            for s, v in best_specs:
                lines.append(f"- {s} (score={v:.2f})")
        else:
            lines.append("")
            lines.append("🧭 **Suggested specialty (from top cases):**")
            for s, v in best_specs:
                lines.append(f"- {s} (score={v:.2f})")

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
    lines.append("⚠️ Это учебный помощник, не медицинский диагноз." if lang == "ru"
                 else "⚠️ Educational assistant, not a medical diagnosis.")

    return "\n".join(lines)