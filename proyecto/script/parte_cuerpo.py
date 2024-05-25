from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
with open("../datos_json/datos_ejercicios_submuestreo.json", "r") as file:
    datos = json.load(file)


def clean_description(description):
    cleaned_description = description.lower()
    return cleaned_description



def find_similar_exercises(body_part, top_n=10):

    buscar_ejercicios = [exercise for exercise in datos if exercise["BodyPart"].lower() == body_part.lower()]
    if not buscar_ejercicios:
        print(f"No se encontraron ejercicios para la parte del cuerpo '{body_part}' en el conjunto de datos.")
        return []

    descripcion_ejercicio = [exercise["Desc"] for exercise in buscar_ejercicios]
    exercise_embeddings = model.encode(descripcion_ejercicio, convert_to_tensor=True)
    
    def cosine_similarity(query_embedding):
        similarities = util.pytorch_cos_sim(query_embedding, exercise_embeddings)[0]
        similar_indices = similarities.argsort(descending=True)
        similar_exercises = []
        for i in similar_indices:
            title = buscar_ejercicios[i]["Title"]
            description = buscar_ejercicios[i]["Desc"]
            body_part = buscar_ejercicios[i]['BodyPart']
            similarity = similarities[i].item()
            if title not in [ex[0] for ex in similar_exercises]:
                similar_exercises.append((title, description, similarity,body_part))
            if len(similar_exercises) >= top_n:
                break
        return similar_exercises

    return cosine_similarity




