import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Cargar los datos de ejercicios desde un archivo JSON
with open("../datos_json/datos_ejercicios_submuestreo.json", "r", encoding="utf-8") as file:
    exercise_data = json.load(file)

# Cargar el modelo preentrenado de Sentence Transformers
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

def find_description_by_title(title):
    for exercise in exercise_data:
        if exercise["Title"].lower() == title.lower():
            return exercise["Desc"]
    return None

def find_similar_exercises_by_title(title, n=5):
    query_description = find_description_by_title(title)
    if query_description is None:
        print("El ejercicio no fue encontrado.")
        return []

    descriptions = [query_description] + [exercise["Desc"] for exercise in exercise_data if exercise["Title"].lower() != title.lower()]
    body_parts = [exercise["BodyPart"] for exercise in exercise_data if exercise["Title"].lower() != title.lower()]
    embeddings = model.encode(descriptions)

    query_embedding = embeddings[0].reshape(1, -1)
    description_embeddings = embeddings[1:]

    similarities = cosine_similarity(query_embedding, description_embeddings)[0]

    similar_indices = similarities.argsort()[::-1][:n]
    similar_exercises = [(exercise_data[i+1]["Title"], descriptions[i], body_parts[i], similarities[i]) for i in similar_indices]

    return similar_exercises

