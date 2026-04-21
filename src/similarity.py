import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import re

from knowledge_graph import SYNONYMS_RU_TO_EN


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mtsamples.csv")


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


class CaseSimilarity:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

        self.df = self.df[["sample_name", "description"]].dropna()

        # ускоряем
        self.df = self.df.head(2000)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        self.matrix = self.vectorizer.fit_transform(self.df["description"])

    def preprocess_query(self, text: str) -> str:
        text = normalize_text(text)
        text = apply_ru_to_en(text)
        return text

    def find_similar(self, text: str, top_k: int = 3):
        text = self.preprocess_query(text)

        if not text.strip():
            return []

        query_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            # фильтр мусора
            if score < 0.05:
                continue

            results.append({
                "case": self.df.iloc[idx]["sample_name"],
                "score": round(score * 100, 2)
            })

        return results