import re
from difflib import get_close_matches

from knowledge_graph import SYNONYMS_RU_TO_EN


YES_WORDS = {"да", "есть", "ага", "угу", "конечно", "yes", "y", "yeah", "yep"}
NO_WORDS = {"нет", "неа", "не", "no", "n", "nope"}

NEGATION_WORDS_RU = {
    "не",
    "нет",
    "без",
    "отсутствует",
    "отсутствуют",
    "отрицаю",
    "отрицает",
}

NEGATION_WORDS_EN = {
    "no",
    "not",
    "without",
    "denies",
    "deny",
    "denied",
}

RU_PATTERN_NORMALIZATION = [
    (r"\bтошнит\b", "тошнота"),
    (r"\bподташнивает\b", "тошнота"),
    (r"\bрвет\b", "рвота"),
    (r"\bвырвало\b", "рвота"),
    (r"\bрвёт\b", "рвота"),
    (r"\bкашляю\b", "кашель"),
    (r"\bкашляет\b", "кашель"),
    (r"\bчихаю\b", "чихание"),
    (r"\bтемпературы\b", "температура"),
    (r"\bознобит\b", "озноб"),
    (r"\bслабый\b", "слабость"),
    (r"\bслабая\b", "слабость"),
    (r"\bслабоcть\b", "слабость"),
    (r"\bголова болит\b", "головная боль"),
    (r"\bболит голова\b", "головная боль"),
    (r"\bкружится голова\b", "головокружение"),
    (r"\bгорло болит\b", "боль в горле"),
    (r"\bболит горло\b", "боль в горле"),
    (r"\bв груди болит\b", "боль в груди"),
    (r"\bболит грудь\b", "боль в груди"),
    (r"\bживот болит\b", "боль в животе"),
    (r"\bболит живот\b", "боль в животе"),
    (r"\bболи в животе\b", "боль в животе"),
    (r"\bболей в животе\b", "боль в животе"),
    (r"\bв животе болит\b", "боль в животе"),
    (r"\bспина болит\b", "боль в спине"),
    (r"\bболит спина\b", "боль в спине"),
    (r"\bв пояснице болит\b", "боль в пояснице"),
    (r"\bболит поясница\b", "боль в пояснице"),
    (r"\bболи в пояснице\b", "боль в пояснице"),
    (r"\bболи в суставах\b", "боль в суставах"),
    (r"\bсуставы болят\b", "боль в суставах"),
    (r"\bмышцы болят\b", "боль в мышцах"),
    (r"\bболи в мышцах\b", "боль в мышцах"),
    (r"\bодышка\b", "одышка"),
    (r"\bтрудно дышать\b", "одышка"),
    (r"\bтяжело дышать\b", "одышка"),
    (r"\bнехватка воздуха\b", "одышка"),
    (r"\bизжога\b", "изжога"),
    (r"\bвздутие живота\b", "вздутие"),
    (r"\bсыпало\b", "сыпь"),
    (r"\bсыпет\b", "сыпь"),
    (r"\bзудит\b", "зуд"),
    (r"\bчешется\b", "зуд"),
    (r"\bчасто мочусь\b", "частое мочеиспускание"),
    (r"\bчасто хожу в туалет\b", "частое мочеиспускание"),
    (r"\bбольно мочиться\b", "боль при мочеиспускании"),
    (r"\bжжет при мочеиспускании\b", "жжение при мочеиспускании"),
    (r"\bжжёт при мочеиспускании\b", "жжение при мочеиспускании"),
    (r"\bпамять ухудшилась\b", "потеря памяти"),
    (r"\bвсе забываю\b", "забывчивость"),
    (r"\bвсё забываю\b", "забывчивость"),
    (r"\bспутанность\b", "спутанность сознания"),
    (r"\bклонит в сон\b", "сонливость"),
    (r"\bсухо во рту\b", "сухость во рту"),
    (r"\bпостоянно хочу пить\b", "сильная жажда"),
    (r"\bсильная жажда\b", "сильная жажда"),
    (r"\bразмыто вижу\b", "размытое зрение"),
    (r"\bзрение ухудшилось\b", "ухудшение зрения"),
    (r"\bпожелтела кожа\b", "пожелтение кожи"),
    (r"\bпожелтели глаза\b", "пожелтение кожи"),
    (r"\bотеки ног\b", "отеки ног"),
    (r"\bотеки ступней\b", "отеки ступней"),
    (r"\bсердце колотится\b", "учащенное сердцебиение"),
    (r"\bсердце бьется часто\b", "учащенное сердцебиение"),
    (r"\bне могу уснуть\b", "бессонница"),
]


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("ё", "е")
    text = re.sub(r"[,_;/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_ru_patterns(text: str) -> str:
    normalized = text
    for pattern, replacement in RU_PATTERN_NORMALIZATION:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def apply_ru_to_en(text: str) -> str:
    normalized = text
    for ru, en in sorted(SYNONYMS_RU_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(ru)}\b", en, normalized)
    return normalized


def preprocess_text(text: str) -> str:
    text = normalize_text(text)
    text = normalize_ru_patterns(text)
    text = apply_ru_to_en(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_symptom_nodes(graph):
    return [node for node, data in graph.nodes(data=True) if data.get("type") == "symptom"]


def is_yes(text: str) -> bool:
    return normalize_text(text) in YES_WORDS


def is_no(text: str) -> bool:
    return normalize_text(text) in NO_WORDS


def _tokenize_words(text: str):
    return re.findall(r"[a-zа-я]+", text.lower())


def _contains_negation_before_symptom(text: str, symptom: str, window: int = 3) -> bool:
    words = _tokenize_words(text)
    symptom_words = symptom.lower().split()
    if not words or not symptom_words:
        return False

    negations = NEGATION_WORDS_RU | NEGATION_WORDS_EN
    symptom_len = len(symptom_words)

    for i in range(len(words) - symptom_len + 1):
        if words[i:i + symptom_len] == symptom_words:
            start = max(0, i - window)
            context = words[start:i]
            if any(word in negations for word in context):
                return True

    return False


def _contains_negation_after_symptom(text: str, symptom: str, window: int = 2) -> bool:
    words = _tokenize_words(text)
    symptom_words = symptom.lower().split()
    if not words or not symptom_words:
        return False

    negations = NEGATION_WORDS_RU | NEGATION_WORDS_EN
    symptom_len = len(symptom_words)

    for i in range(len(words) - symptom_len + 1):
        if words[i:i + symptom_len] == symptom_words:
            end = i + symptom_len
            context = words[end:end + window]
            if any(word in negations for word in context):
                return True

    return False


def _contains_negation_pattern(text: str, symptom: str) -> bool:
    neg = r"(?:не|нет|без|отсутствует|отсутствуют|отрицаю|отрицает|no|not|without|denies|deny|denied)"
    symptom_pattern = re.escape(symptom).replace(r"\ ", r"\s+")

    patterns = [
        rf"\b{neg}\b(?:\s+\w+){{0,2}}\s+{symptom_pattern}\b",
        rf"\b{neg}\b\s+{symptom_pattern}\b",
        rf"\b{symptom_pattern}\b(?:\s+\w+){{0,2}}\s+\b{neg}\b",
        rf"\b{symptom_pattern}\b\s+\b{neg}\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def is_negated_symptom(text: str, symptom: str) -> bool:
    normalized_text = normalize_text(text)
    symptom = symptom.lower().strip()
    if not symptom:
        return False

    return (
        _contains_negation_before_symptom(normalized_text, symptom)
        or _contains_negation_after_symptom(normalized_text, symptom)
        or _contains_negation_pattern(normalized_text, symptom)
    )


def extract_symptoms(text: str, graph, fuzzy_cutoff: float = 0.88):
    text = preprocess_text(text)

    symptom_nodes = get_symptom_nodes(graph)
    found = set()
    negated = set()

    padded_text = f" {text} "
    for symptom in sorted(symptom_nodes, key=len, reverse=True):
        if f" {symptom} " in padded_text:
            if is_negated_symptom(text, symptom):
                negated.add(symptom)
            else:
                found.add(symptom)

    tokens = re.findall(r"[a-zа-я]+", text)
    for token in tokens:
        if token in symptom_nodes:
            if is_negated_symptom(text, token):
                negated.add(token)
            else:
                found.add(token)
            continue

        close = get_close_matches(token, symptom_nodes, n=1, cutoff=fuzzy_cutoff)
        if close:
            matched = close[0]
            if is_negated_symptom(text, matched):
                negated.add(matched)
            else:
                found.add(matched)

    found -= negated
    return sorted(found)