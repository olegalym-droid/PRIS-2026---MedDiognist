from collections import Counter

from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MultiLabelBinarizer


SAMPLES_PER_CASE = 80


class SpecialtyMLPredictor:
    def __init__(self, graph):
        self.graph = graph

        self.symptom_binarizer = None
        self.specialty_model = None
        self.case_model = None

        self.is_trained = False
        self.training_summary = {}

        self._build_and_train()

    def _collect_manual_cases(self):
        cases = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") != "case":
                continue
            if data.get("source") != "manual":
                continue

            specialty = (data.get("specialty") or "").strip()
            title_ru = (data.get("title_ru") or "").strip()
            title_en = (data.get("title_en") or "").strip()

            neighbors = list(self.graph.neighbors(node))
            symptoms = sorted(
                n for n in neighbors
                if self.graph.nodes[n].get("type") == "symptom"
            )

            if not specialty or not symptoms:
                continue

            cases.append(
                {
                    "case_id": node,
                    "title_ru": title_ru or node,
                    "title_en": title_en or node,
                    "specialty": specialty,
                    "symptoms": symptoms,
                }
            )
        return cases

    def _generate_augmented_samples(self, cases):
        symptom_sets = []
        specialty_labels = []
        case_labels = []

        for case in cases:
            base_symptoms = case["symptoms"]
            n = len(base_symptoms)

            for i in range(SAMPLES_PER_CASE):
                chosen = set()

                # базовый полный пример
                if i == 0:
                    chosen.update(base_symptoms)
                else:
                    # берем подмножества симптомов сценария
                    keep_count = max(1, min(n, 1 + (i % n)))
                    chosen.update(base_symptoms[:keep_count])

                    # добавляем еще один вариант со всеми симптомами
                    if i % 5 == 0:
                        chosen.update(base_symptoms)

                symptom_sets.append(sorted(chosen))
                specialty_labels.append(case["specialty"])
                case_labels.append(case["case_id"])

        return symptom_sets, specialty_labels, case_labels

    def _build_and_train(self):
        cases = self._collect_manual_cases()
        if len(cases) < 2:
            self.is_trained = False
            return

        symptom_sets, specialty_labels, case_labels = self._generate_augmented_samples(cases)

        self.symptom_binarizer = MultiLabelBinarizer()
        X = self.symptom_binarizer.fit_transform(symptom_sets)

        if len(X) == 0:
            self.is_trained = False
            return

        if len(set(specialty_labels)) < 2:
            self.is_trained = False
            return

        if len(set(case_labels)) < 2:
            self.is_trained = False
            return

        self.specialty_model = BernoulliNB()
        self.specialty_model.fit(X, specialty_labels)

        self.case_model = BernoulliNB()
        self.case_model.fit(X, case_labels)

        specialty_counts = Counter(specialty_labels)
        case_counts = Counter(case_labels)

        case_titles = {}
        for case in cases:
            case_titles[case["case_id"]] = {
                "title_ru": case["title_ru"],
                "title_en": case["title_en"],
                "specialty": case["specialty"],
                "symptoms": case["symptoms"],
            }

        self.training_summary = {
            "num_cases": len(cases),
            "num_samples": len(symptom_sets),
            "num_features": len(self.symptom_binarizer.classes_),
            "specialty_counts": dict(sorted(specialty_counts.items())),
            "case_counts": dict(sorted(case_counts.items())),
            "case_titles": case_titles,
        }

        self.is_trained = True

    def _vectorize(self, symptoms):
        known = set(self.symptom_binarizer.classes_)
        cleaned = [s for s in symptoms if s in known]
        return self.symptom_binarizer.transform([cleaned])

    def predict_top_specialties(self, symptoms, top_k=3):
        if not self.is_trained or not symptoms:
            return []

        X = self._vectorize(symptoms)
        probs = self.specialty_model.predict_proba(X)[0]
        classes = list(self.specialty_model.classes_)

        ranked = sorted(
            zip(classes, probs),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []
        for specialty, prob in ranked[:top_k]:
            results.append(
                {
                    "specialty": specialty,
                    "probability": round(float(prob) * 100, 2),
                    "training_cases": self.training_summary["specialty_counts"].get(specialty, 0),
                }
            )
        return results

    def predict_top_cases(self, symptoms, top_k=3):
        if not self.is_trained or not symptoms:
            return []

        X = self._vectorize(symptoms)
        probs = self.case_model.predict_proba(X)[0]
        classes = list(self.case_model.classes_)

        ranked = sorted(
            zip(classes, probs),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []
        for case_id, prob in ranked[:top_k]:
            meta = self.training_summary["case_titles"].get(case_id, {})
            results.append(
                {
                    "case_id": case_id,
                    "title_ru": meta.get("title_ru", case_id),
                    "title_en": meta.get("title_en", case_id),
                    "specialty": meta.get("specialty", ""),
                    "probability": round(float(prob) * 100, 2),
                    "symptoms": meta.get("symptoms", []),
                    "training_cases": self.training_summary["case_counts"].get(case_id, 0),
                }
            )
        return results

    def get_training_summary(self):
        return {
            "is_trained": self.is_trained,
            "num_cases": self.training_summary.get("num_cases", 0),
            "num_samples": self.training_summary.get("num_samples", 0),
            "num_features": self.training_summary.get("num_features", 0),
            "specialty_counts": self.training_summary.get("specialty_counts", {}),
        }