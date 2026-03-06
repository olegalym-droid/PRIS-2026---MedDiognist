import os
import re
import math
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mtsamples.csv")

# RU -> EN (канонические symptom-термины)
SYNONYMS_RU_TO_EN = {
    "температура": "fever",
    "высокая температура": "fever",
    "озноб": "chills",
    "слабость": "fatigue",
    "головная боль": "headache",
    "головокружение": "dizziness",
    "кашель": "cough",
    "сухой кашель": "dry cough",
    "влажный кашель": "productive cough",
    "насморк": "rhinorrhea",
    "заложенность носа": "nasal congestion",
    "чихание": "sneezing",
    "боль в горле": "sore throat",
    "одышка": "shortness of breath",
    "боль в груди": "chest pain",
    "тошнота": "nausea",
    "рвота": "vomiting",
    "понос": "diarrhea",
    "жидкий стул": "diarrhea",
    "запор": "constipation",
    "боль в животе": "abdominal pain",
    "изжога": "heartburn",
    "вздутие": "bloating",
    "боль в пояснице": "flank pain",
    "боль в спине": "back pain",
    "боль в суставах": "joint pain",
    "боль в мышцах": "muscle pain",
    "онемение": "numbness",
    "судороги": "seizure",
    "потеря сознания": "loss of consciousness",
    "кровотечение": "bleeding",
    "кровь в моче": "hematuria",
    "кровь в стуле": "blood in stool",
    "кровь из носа": "epistaxis",
    "носовое кровотечение": "epistaxis",
    "сыпь": "rash",
    "зуд": "itching",
    "покраснение кожи": "erythema",
    "боль при мочеиспускании": "dysuria",
    "частое мочеиспускание": "urinary frequency",
    "жжение при мочеиспускании": "dysuria",
    "стрельба в ушах": "ear pain",
    "боль в ухе": "ear pain",
    "шум в ушах": "tinnitus",
    "потеря слуха": "hearing loss",
}

# EN -> RU (для красивого вывода)
DISPLAY_RU = {v: k for k, v in SYNONYMS_RU_TO_EN.items()}
DISPLAY_RU.update({
    "productive cough": "влажный кашель",
    "dry cough": "сухой кашель",
    "nasal congestion": "заложенность носа",
    "rhinorrhea": "насморк",
    "shortness of breath": "одышка",
    "loss of consciousness": "потеря сознания",
    "blood in stool": "кровь в стуле",
})

# Симптомный словарь (whitelist). Мы оставляем только эти keywords как "симптомы".
SYMPTOM_LEXICON_EN = {
    "fever", "chills", "fatigue", "weakness",
    "headache", "dizziness",
    "cough", "dry cough", "productive cough",
    "rhinorrhea", "nasal congestion", "sneezing", "sore throat",
    "shortness of breath", "chest pain",
    "nausea", "vomiting", "diarrhea", "constipation",
    "abdominal pain", "heartburn", "bloating",
    "back pain", "flank pain", "joint pain", "muscle pain",
    "rash", "itching", "erythema",
    "bleeding", "epistaxis", "hematuria", "blood in stool",
    "dysuria", "urinary frequency",
    "numbness", "seizure", "loss of consciousness",
    "ear pain", "tinnitus", "hearing loss",
}

# Доп. правила: некоторые keywords в mtsamples будут вида "pain", "pain in chest" и т.п.
SYMPTOM_PATTERNS = [
    r"\bpain\b",
    r"\bfever\b",
    r"\bcough\b",
    r"\bnausea\b",
    r"\bvomit",
    r"\bdiarrhea\b",
    r"\bshortness of breath\b",
    r"\bbleed",
    r"\brash\b",
    r"\bdysuria\b",
    r"\bhematuria\b",
]

# Шум, который убираем
STOP_KEYWORDS = {
    "surgery", "operative", "procedure", "biopsy", "imaging", "radiology",
    "history and physical", "consult", "discharge summary",
    "laparoscopic", "anesthesia", "follow-up", "clinic",
    "diagnosis", "assessment", "plan", "examination",
}

def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _split_keywords(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    parts = [p.strip() for p in str(s).split(",")]
    parts = [_norm(p) for p in parts if p.strip()]
    return parts

def _is_symptom_keyword(kw: str) -> bool:
    if not kw or len(kw) > 80:
        return False
    if kw in STOP_KEYWORDS:
        return False
    if kw in SYMPTOM_LEXICON_EN:
        return True
    for pat in SYMPTOM_PATTERNS:
        if re.search(pat, kw):
            return True
    return False

def create_graph_from_mtsamples(csv_path: str = DATA_PATH, max_cases: int | None = None):
    df = pd.read_csv(csv_path)
    if max_cases is not None:
        df = df.head(max_cases)

    # собираем keywords -> только симптомные
    case_to_keywords = {}
    df_counts = Counter()

    for _, row in df.iterrows():
        case = _norm(row.get("sample_name", ""))
        if not case:
            continue

        kws = _split_keywords(row.get("keywords", ""))
        kws = [k for k in kws if _is_symptom_keyword(k)]
        kws = list(dict.fromkeys(kws))  # уникальные, сохранить порядок

        case_to_keywords[case] = kws
        for k in kws:
            df_counts[k] += 1

    total_cases = max(1, len(case_to_keywords))

    G = nx.Graph()

    # symptom-keyword nodes with idf
    for kw, dfi in df_counts.items():
        idf = math.log((total_cases + 1) / (dfi + 1)) + 1.0
        G.add_node(kw, type="symptom", df=dfi, idf=idf, ru=DISPLAY_RU.get(kw, ""))

    # case + specialty nodes
    for _, row in df.iterrows():
        case = _norm(row.get("sample_name", ""))
        if not case or case not in case_to_keywords:
            continue

        spec = _norm(row.get("medical_specialty", ""))
        desc = str(row.get("description", "")) if row.get("description") is not None else ""

        G.add_node(case, type="case", specialty=spec, description=desc)

        if spec:
            G.add_node(spec, type="specialty")
            G.add_edge(case, spec)

        for kw in case_to_keywords[case]:
            if kw in G.nodes:
                G.add_edge(case, kw)

    return G

def load_graph():
    if os.path.exists(DATA_PATH):
        return create_graph_from_mtsamples(DATA_PATH)
    raise FileNotFoundError(f"Не найден датасет: {DATA_PATH}. Положи mtsamples.csv в data/raw/")