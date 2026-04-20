import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "mtsamples.csv")


class CaseSimilarity:
    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)

        # берем только описание
        self.df = self.df[["sample_name", "description"]].dropna()

        # уменьшаем размер (ускорение)
        self.df = self.df.head(2000)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        self.matrix = self.vectorizer.fit_transform(self.df["description"])

    def find_similar(self, text: str, top_k: int = 3):
        if not text.strip():
            return []

        query_vec = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_vec, self.matrix)[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "case": self.df.iloc[idx]["sample_name"],
                "score": round(float(similarities[idx]) * 100, 2)
            })

        return results