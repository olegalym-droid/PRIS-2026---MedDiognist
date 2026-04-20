import re
from difflib import get_close_matches

from knowledge_graph import SYNONYMS_RU_TO_EN


YES_WORDS = {"да", "есть", "ага", "угу", "конечно", "yes", "y", "yeah", "yep"}
NO_WORDS = {"нет", "неа", "не", "no", "n", "nope"}


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


def is_yes(text: str) -> bool:
    return normalize_text(text) in YES_WORDS


def is_no(text: str) -> bool:
    return normalize_text(text) in NO_WORDS