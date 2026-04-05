# src/knowledge_graph.py

import os
import re
import math
from collections import Counter

import pandas as pd
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mtsamples.csv")

SYNONYMS_RU_TO_EN = {
    "высокая температура": "fever",
    "температура": "fever",
    "жар": "fever",
    "озноб": "chills",
    "слабость": "weakness",
    "усталость": "fatigue",
    "быстрая утомляемость": "fatigue",
    "головная боль": "headache",
    "головокружение": "dizziness",
    "обморок": "loss of consciousness",
    "сухой кашель": "dry cough",
    "влажный кашель": "productive cough",
    "кашель": "cough",
    "насморк": "rhinorrhea",
    "заложенность носа": "nasal congestion",
    "чихание": "sneezing",
    "боль в горле": "sore throat",
    "одышка": "shortness of breath",
    "затрудненное дыхание": "shortness of breath",
    "боль в груди": "chest pain",
    "тошнота": "nausea",
    "рвота": "vomiting",
    "понос": "diarrhea",
    "диарея": "diarrhea",
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
    "жжение при мочеиспускании": "dysuria",
    "частое мочеиспускание": "urinary frequency",
    "боль в ухе": "ear pain",
    "шум в ушах": "tinnitus",
    "потеря слуха": "hearing loss",
    "потеря памяти": "memory loss",
    "забывчивость": "forgetfulness",
    "спутанность сознания": "confusion",
    "сонливость": "drowsiness",
    "сухость во рту": "dry mouth",
    "сильная жажда": "excessive thirst",
    "жажда": "excessive thirst",
    "частый голод": "increased appetite",
    "размытое зрение": "blurred vision",
    "ухудшение зрения": "vision loss",
    "пожелтение кожи": "jaundice",
    "желтушность": "jaundice",
    "отеки": "edema",
    "отеки ног": "leg swelling",
    "отеки ступней": "leg swelling",
    "потеря веса": "weight loss",
    "набор веса": "weight gain",
    "дрожь": "tremor",
    "тремор": "tremor",
    "учащенное сердцебиение": "palpitations",
    "сердцебиение": "palpitations",
    "нарушение сна": "insomnia",
    "бессонница": "insomnia",
    "тревога": "anxiety",
    "депрессия": "depressed mood",
    "подавленность": "depressed mood",
}

DISPLAY_RU = {v: k for k, v in SYNONYMS_RU_TO_EN.items()}
DISPLAY_RU.update(
    {
        "productive cough": "влажный кашель",
        "dry cough": "сухой кашель",
        "nasal congestion": "заложенность носа",
        "rhinorrhea": "насморк",
        "shortness of breath": "одышка",
        "loss of consciousness": "потеря сознания",
        "blood in stool": "кровь в стуле",
        "urinary frequency": "частое мочеиспускание",
        "memory loss": "потеря памяти",
        "forgetfulness": "забывчивость",
        "confusion": "спутанность сознания",
        "dry mouth": "сухость во рту",
        "excessive thirst": "сильная жажда",
        "increased appetite": "частый голод",
        "blurred vision": "размытое зрение",
        "vision loss": "ухудшение зрения",
        "jaundice": "пожелтение кожи",
        "leg swelling": "отеки ног",
        "weight loss": "потеря веса",
        "weight gain": "набор веса",
        "tremor": "дрожь",
        "palpitations": "учащенное сердцебиение",
        "insomnia": "бессонница",
        "depressed mood": "подавленность",
    }
)

SPECIALTY_DISPLAY_RU = {
    "pulmonary": "Пульмонолог",
    "cardiology": "Кардиолог",
    "neurology": "Невролог",
    "gastroenterology": "Гастроэнтеролог",
    "urology": "Уролог",
    "otolaryngology": "ЛОР",
    "orthopedic": "Ортопед",
    "orthopaedic": "Ортопед",
    "allergy / immunology": "Аллерголог / иммунолог",
    "allergy/immunology": "Аллерголог / иммунолог",
    "internal medicine": "Терапевт",
    "emergency room reports": "Неотложная помощь",
    "emergency medicine": "Неотложная помощь",
    "family medicine": "Семейный врач",
    "general medicine": "Терапевт",
    "neurosurgery": "Нейрохирург",
    "pain management": "Специалист по лечению боли",
    "pediatrics": "Педиатр",
    "psychiatry": "Психиатр",
    "radiology": "Радиолог",
    "rheumatology": "Ревматолог",
    "surgery": "Хирург",
    "dermatology": "Дерматолог",
    "infectious disease": "Инфекционист",
    "infectious diseases": "Инфекционист",
    "endocrinology": "Эндокринолог",
    "nephrology": "Нефролог",
    "oncology": "Онколог",
    "ophthalmology": "Офтальмолог",
    "obstetrics / gynecology": "Акушер-гинеколог",
    "obstetrics/gynecology": "Акушер-гинеколог",
    "hepatology": "Гепатолог",
}

MANUAL_CASES = [
    {
        "id": "acute_viral_upper_respiratory_infection",
        "title_en": "Acute viral upper respiratory infection",
        "title_ru": "Острая вирусная инфекция верхних дыхательных путей",
        "specialty": "internal medicine",
        "keywords": ["fever", "cough", "sore throat", "rhinorrhea", "nasal congestion", "sneezing", "fatigue"],
    },
    {
        "id": "bronchitis_or_lower_respiratory_infection",
        "title_en": "Bronchitis or lower respiratory infection",
        "title_ru": "Бронхит или инфекция нижних дыхательных путей",
        "specialty": "pulmonary",
        "keywords": ["cough", "productive cough", "fever", "shortness of breath", "fatigue", "chills"],
    },
    {
        "id": "gastroenteritis",
        "title_en": "Gastroenteritis",
        "title_ru": "Гастроэнтерит",
        "specialty": "gastroenterology",
        "keywords": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever", "weakness"],
    },
    {
        "id": "urinary_tract_infection",
        "title_en": "Urinary tract infection",
        "title_ru": "Инфекция мочевыводящих путей",
        "specialty": "urology",
        "keywords": ["dysuria", "urinary frequency", "fever", "hematuria", "abdominal pain", "back pain"],
    },
    {
        "id": "migraine_or_headache_syndrome",
        "title_en": "Migraine or headache syndrome",
        "title_ru": "Мигрень или головная боль",
        "specialty": "neurology",
        "keywords": ["headache", "dizziness", "nausea", "vision loss"],
    },
    {
        "id": "allergic_condition",
        "title_en": "Allergic condition",
        "title_ru": "Аллергическое состояние",
        "specialty": "allergy / immunology",
        "keywords": ["rash", "itching", "sneezing", "rhinorrhea", "nasal congestion", "erythema"],
    },
    {
        "id": "possible_diabetes_mellitus",
        "title_en": "Possible diabetes mellitus",
        "title_ru": "Возможный сахарный диабет",
        "specialty": "endocrinology",
        "keywords": ["excessive thirst", "dry mouth", "urinary frequency", "weight loss", "blurred vision", "fatigue", "increased appetite"],
    },
    {
        "id": "cognitive_impairment_or_dementia",
        "title_en": "Cognitive impairment or dementia",
        "title_ru": "Когнитивное снижение или деменция",
        "specialty": "neurology",
        "keywords": ["memory loss", "forgetfulness", "confusion", "drowsiness"],
    },
    {
        "id": "possible_liver_disorder",
        "title_en": "Possible liver disorder",
        "title_ru": "Возможное заболевание печени",
        "specialty": "hepatology",
        "keywords": ["jaundice", "abdominal pain", "nausea", "fatigue", "weight loss", "itching"],
    },
    {
        "id": "visual_disturbance",
        "title_en": "Visual disturbance",
        "title_ru": "Нарушение зрения",
        "specialty": "ophthalmology",
        "keywords": ["blurred vision", "vision loss", "headache", "dizziness"],
    },
    {
        "id": "heart_rhythm_or_cardiac_symptoms",
        "title_en": "Heart rhythm or cardiac symptoms",
        "title_ru": "Нарушение ритма или сердечные симптомы",
        "specialty": "cardiology",
        "keywords": ["chest pain", "shortness of breath", "palpitations", "dizziness", "weakness", "leg swelling"],
    },
    {
        "id": "thyroid_or_metabolic_disorder",
        "title_en": "Thyroid or metabolic disorder",
        "title_ru": "Возможное нарушение щитовидной железы или обмена веществ",
        "specialty": "endocrinology",
        "keywords": ["weight loss", "weight gain", "fatigue", "tremor", "palpitations", "anxiety", "insomnia"],
    },
]

SYMPTOM_LEXICON_EN = {
    "fever",
    "chills",
    "fatigue",
    "weakness",
    "headache",
    "dizziness",
    "cough",
    "dry cough",
    "productive cough",
    "rhinorrhea",
    "nasal congestion",
    "sneezing",
    "sore throat",
    "shortness of breath",
    "chest pain",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "abdominal pain",
    "heartburn",
    "bloating",
    "back pain",
    "flank pain",
    "joint pain",
    "muscle pain",
    "rash",
    "itching",
    "erythema",
    "bleeding",
    "epistaxis",
    "hematuria",
    "blood in stool",
    "dysuria",
    "urinary frequency",
    "numbness",
    "seizure",
    "loss of consciousness",
    "ear pain",
    "tinnitus",
    "hearing loss",
    "memory loss",
    "forgetfulness",
    "confusion",
    "drowsiness",
    "dry mouth",
    "excessive thirst",
    "increased appetite",
    "blurred vision",
    "vision loss",
    "jaundice",
    "edema",
    "leg swelling",
    "weight loss",
    "weight gain",
    "tremor",
    "palpitations",
    "insomnia",
    "anxiety",
    "depressed mood",
}

SYMPTOM_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\babdominal pain\b",
    r"\bback pain\b",
    r"\bflank pain\b",
    r"\bjoint pain\b",
    r"\bmuscle pain\b",
    r"\bsore throat\b",
    r"\bear pain\b",
    r"\bheadache\b",
    r"\bdizziness\b",
    r"\bfever\b",
    r"\bcough\b",
    r"\bnausea\b",
    r"\bvomit",
    r"\bdiarrhea\b",
    r"\bconstipation\b",
    r"\bbleed",
    r"\brash\b",
    r"\bitch",
    r"\bdysuria\b",
    r"\bhematuria\b",
    r"\btinnitus\b",
    r"\bhearing loss\b",
    r"\bmemory loss\b",
    r"\bforgetfulness\b",
    r"\bconfusion\b",
    r"\bdry mouth\b",
    r"\bexcessive thirst\b",
    r"\bblurred vision\b",
    r"\bvision loss\b",
    r"\bjaundice\b",
    r"\bleg swelling\b",
    r"\bweight loss\b",
    r"\bweight gain\b",
    r"\btremor\b",
    r"\bpalpitations\b",
    r"\binsomnia\b",
    r"\banxiety\b",
]

STOP_KEYWORDS = {
    "surgery",
    "operative",
    "procedure",
    "biopsy",
    "imaging",
    "radiology",
    "history and physical",
    "consult",
    "discharge summary",
    "laparoscopic",
    "anesthesia",
    "follow-up",
    "clinic",
    "diagnosis",
    "assessment",
    "plan",
    "examination",
    "normal exam",
    "dictation",
    "colonoscopy",
    "esophagogastroduodenoscopy",
    "biopsy",
}

TITLE_PHRASE_MAP_RU = {
    "shortness of breath": "одышка",
    "chest pain": "боль в груди",
    "abdominal pain": "боль в животе",
    "back pain": "боль в спине",
    "flank pain": "боль в пояснице",
    "sore throat": "боль в горле",
    "ear pain": "боль в ухе",
    "hearing loss": "потеря слуха",
    "blood in stool": "кровь в стуле",
    "loss of consciousness": "потеря сознания",
    "urinary frequency": "частое мочеиспускание",
    "dry cough": "сухой кашель",
    "productive cough": "влажный кашель",
    "nasal congestion": "заложенность носа",
    "rhinorrhea": "насморк",
    "headache": "головная боль",
    "dizziness": "головокружение",
    "nausea": "тошнота",
    "vomiting": "рвота",
    "diarrhea": "диарея",
    "constipation": "запор",
    "bloating": "вздутие",
    "heartburn": "изжога",
    "fever": "температура",
    "chills": "озноб",
    "fatigue": "усталость",
    "weakness": "слабость",
    "cough": "кашель",
    "sneezing": "чихание",
    "rash": "сыпь",
    "itching": "зуд",
    "erythema": "покраснение",
    "bleeding": "кровотечение",
    "epistaxis": "кровь из носа",
    "hematuria": "кровь в моче",
    "dysuria": "боль при мочеиспускании",
    "numbness": "онемение",
    "seizure": "судороги",
    "tinnitus": "шум в ушах",
    "memory loss": "потеря памяти",
    "forgetfulness": "забывчивость",
    "confusion": "спутанность сознания",
    "blurred vision": "размытое зрение",
    "vision loss": "ухудшение зрения",
    "jaundice": "желтуха",
    "palpitations": "сердцебиение",
}

TITLE_TOKEN_MAP_RU = {
    "acute": "острый",
    "chronic": "хронический",
    "viral": "вирусный",
    "bacterial": "бактериальный",
    "upper": "верхний",
    "lower": "нижний",
    "respiratory": "респираторный",
    "infection": "инфекция",
    "pain": "боль",
    "fever": "температура",
    "cough": "кашель",
    "headache": "головная боль",
    "dizziness": "головокружение",
    "nausea": "тошнота",
    "vomiting": "рвота",
    "abdominal": "абдоминальный",
    "chest": "грудной",
    "throat": "горло",
    "ear": "ухо",
    "nose": "нос",
    "urinary": "мочевой",
    "hearing": "слух",
    "loss": "потеря",
    "shortness": "затруднение",
    "breath": "дыхания",
    "bleeding": "кровотечение",
    "rash": "сыпь",
    "syncope": "обморок",
    "seizure": "судороги",
    "evaluation": "оценка",
    "followup": "контроль",
    "follow-up": "контроль",
    "consultation": "консультация",
    "report": "случай",
    "case": "случай",
}

def _norm(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _split_keywords(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parts = [part.strip() for part in str(value).split(",")]
    return [_norm(part) for part in parts if str(part).strip()]


def _is_symptom_keyword(keyword: str) -> bool:
    if not keyword or len(keyword) > 80:
        return False
    if keyword in STOP_KEYWORDS:
        return False
    if keyword in SYMPTOM_LEXICON_EN:
        return True
    return any(re.search(pattern, keyword) for pattern in SYMPTOM_PATTERNS)


def format_specialty_name(specialty: str, lang: str = "ru") -> str:
    specialty = _norm(specialty)
    if not specialty:
        return ""
    if lang == "en":
        return specialty.title()
    return SPECIALTY_DISPLAY_RU.get(specialty, specialty.title())


def format_case_title(title: str, lang: str = "ru") -> str:
    raw = _norm(title)
    if not raw:
        return ""

    if lang == "en":
        return raw.title()

    text = f" {raw} "
    for src in sorted(TITLE_PHRASE_MAP_RU.keys(), key=len, reverse=True):
        text = text.replace(f" {src} ", f" {TITLE_PHRASE_MAP_RU[src]} ")

    words = []
    for token in text.strip().split():
        words.append(TITLE_TOKEN_MAP_RU.get(token, token))

    result = " ".join(words)
    result = re.sub(r"\s+", " ", result).strip()
    if not result:
        return raw.title()
    return result[:1].upper() + result[1:]


def _add_case_node(graph, case_id: str, title_en: str, title_ru: str, specialty: str, keywords: list[str]):
    graph.add_node(
        case_id,
        type="case",
        specialty=specialty,
        specialty_ru=format_specialty_name(specialty, "ru"),
        title_en=title_en,
        title_ru=title_ru,
        source="manual",
        description="",
    )

    if specialty:
        graph.add_node(
            specialty,
            type="specialty",
            title_en=format_specialty_name(specialty, "en"),
            title_ru=format_specialty_name(specialty, "ru"),
        )
        graph.add_edge(case_id, specialty)

    for kw in keywords:
        if kw not in graph.nodes:
            graph.add_node(kw, type="symptom", df=1, idf=2.0, ru=DISPLAY_RU.get(kw, ""))
        graph.add_edge(case_id, kw)


def create_graph_from_mtsamples(csv_path: str = DATA_PATH, max_cases: int | None = None):
    df = pd.read_csv(csv_path)
    if max_cases is not None:
        df = df.head(max_cases)

    case_to_keywords = {}
    df_counts = Counter()

    for _, row in df.iterrows():
        case = _norm(row.get("sample_name", ""))
        if not case:
            continue

        keywords = _split_keywords(row.get("keywords", ""))
        keywords = [kw for kw in keywords if _is_symptom_keyword(kw)]
        keywords = list(dict.fromkeys(keywords))

        if not keywords:
            continue

        case_to_keywords[case] = keywords
        for kw in keywords:
            df_counts[kw] += 1

    for item in MANUAL_CASES:
        for kw in item["keywords"]:
            df_counts[kw] += 1

    total_cases = max(1, len(case_to_keywords) + len(MANUAL_CASES))
    graph = nx.Graph()

    for kw, doc_freq in df_counts.items():
        idf = math.log((total_cases + 1) / (doc_freq + 1)) + 1.0
        graph.add_node(
            kw,
            type="symptom",
            df=doc_freq,
            idf=idf,
            ru=DISPLAY_RU.get(kw, ""),
        )

    for item in MANUAL_CASES:
        _add_case_node(
            graph,
            case_id=item["id"],
            title_en=item["title_en"],
            title_ru=item["title_ru"],
            specialty=item["specialty"],
            keywords=item["keywords"],
        )

    for _, row in df.iterrows():
        case = _norm(row.get("sample_name", ""))
        if not case or case not in case_to_keywords:
            continue

        specialty = _norm(row.get("medical_specialty", ""))
        description = str(row.get("description", "") or "")

        graph.add_node(
            case,
            type="case",
            specialty=specialty,
            specialty_ru=format_specialty_name(specialty, "ru"),
            title_en=format_case_title(case, "en"),
            title_ru=format_case_title(case, "ru"),
            source="dataset",
            description=description,
        )

        if specialty:
            graph.add_node(
                specialty,
                type="specialty",
                title_en=format_specialty_name(specialty, "en"),
                title_ru=format_specialty_name(specialty, "ru"),
            )
            graph.add_edge(case, specialty)

        for kw in case_to_keywords[case]:
            if kw in graph.nodes:
                graph.add_edge(case, kw)

    return graph


def load_graph():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return create_graph_from_mtsamples(DATA_PATH)