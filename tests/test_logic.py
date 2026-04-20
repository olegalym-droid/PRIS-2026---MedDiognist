import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from knowledge_graph import load_graph
from logic import analyze_case


def make_patient():
    return {
        "confirmed": [],
        "denied": [],
        "pending_question": None,
        "emergency_triggered": False,
        "emergency_term": None,
        "triage": None,
    }


def test_extract_common_respiratory_case():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "кашель, температура, боль в горле",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] in {"ok", "no_cases"}
    assert "кашель" in result["recognized_symptoms"]
    assert "температура" in result["recognized_symptoms"]
    assert result["triage"] in {"MEDIUM", "HIGH", "CRITICAL", "LOW"}


def test_chest_pain_is_critical():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "боль в груди",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] == "emergency"
    assert result["triage"] == "CRITICAL"
    assert result["emergency_term"] == "chest pain"


def test_shortness_of_breath_is_critical():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "одышка",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] == "emergency"
    assert result["triage"] == "CRITICAL"
    assert result["emergency_term"] == "shortness of breath"


def test_possible_diabetes_direction():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "жажда, частое мочеиспускание, размытое зрение",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] in {"ok", "no_cases"}
    assert result["triage"] in {"MEDIUM", "HIGH"}

    specialty_names = [item["name"] for item in result.get("specialties", [])]
    assert "Эндокринолог" in specialty_names or any("эндокрин" in name.lower() for name in specialty_names)

    case_names = [item["case"] for item in result.get("cases", [])]
    assert any("диабет" in case.lower() for case in case_names)


def test_memory_related_case_high_risk():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "потеря памяти, забывчивость, спутанность сознания",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] in {"ok", "emergency", "no_cases"}
    assert result["triage"] in {"HIGH", "CRITICAL"}

    recognized = result["recognized_symptoms"]
    assert "потеря памяти" in recognized
    assert "забывчивость" in recognized


def test_reset_command():
    graph = load_graph()
    patient = make_patient()

    result = analyze_case(
        "/reset",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert result["action"] == "reset"


def test_yes_answer_confirms_pending_symptom():
    graph = load_graph()
    patient = make_patient()
    patient["pending_question"] = "fever"

    result = analyze_case(
        "да",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert "температура" in result["recognized_symptoms"]


def test_no_answer_denies_pending_symptom():
    graph = load_graph()
    patient = make_patient()
    patient["pending_question"] = "fever"

    analyze_case(
        "нет",
        graph,
        patient,
        lang="ru",
        mutate=True,
    )

    assert "fever" in patient["denied"]
    assert patient["pending_question"] is None