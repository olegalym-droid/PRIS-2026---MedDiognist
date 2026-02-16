import networkx as nx
from models import Disease

def create_graph():
    G = nx.Graph()


    flu = Disease(
        name="Грипп",
        symptoms=["температура", "кашель", "слабость"],
        medicines=["Парацетамол", "Ибупрофен"]
    )

    cold = Disease(
        name="Простуда",
        symptoms=["насморк", "кашель", "боль в горле"],
        medicines=["Називин", "Стрепсилс"]
    )

    covid = Disease(
        name="COVID-19",
        symptoms=["температура", "кашель", "одышка"],
        medicines=["Парацетамол", "Антикоагулянты"]
    )

    diseases = [flu, cold, covid]


    for disease in diseases:
        G.add_node(disease.name, type="disease")

        # Добавляем симптомы
        for symptom in disease.symptoms:
            G.add_node(symptom, type="symptom")
            G.add_edge(disease.name, symptom)

        # Добавляем лекарства
        for med in disease.medicines:
            G.add_node(med, type="medicine")
            G.add_edge(disease.name, med)

    return G


def find_related(graph, node):
    if node not in graph:
        return []

    return list(graph.neighbors(node))

def find_diseases_by_symptoms(graph, input_symptoms):
    """
    Возвращает болезни, которые имеют совпадения с симптомами пользователя
    """
    possible_diseases = []

    for node, data in graph.nodes(data=True):
        if data.get("type") == "disease":
            disease_symptoms = list(graph.neighbors(node))

            # считаем совпадения
            matches = set(input_symptoms) & set(disease_symptoms)

            if len(matches) >= 2:  # минимум 2 совпадения
                possible_diseases.append(node)

    return possible_diseases
def load_graph():
    return create_graph()
