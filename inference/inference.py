from sentence_transformers import SentenceTransformer, util
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar los datos de ejercicios desde un archivo JSON
with open("../lab/json_data/datos_ejercicios_submuestreo.json", "r") as file:
    exercise_data = json.load(file)

# Cargar el modelo preentrenado de Sentence Transformers
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def clean_description(description):
    """Preprocesamiento: Limpia la descripción del ejercicio."""
    cleaned_description = description.lower()
    return cleaned_description

def find_similar_exercises(body_part, top_n=10):
    """
    Utilización del modelo: Encuentra ejercicios similares para una parte del cuerpo dada.
    
    Args:
        body_part (str): Parte del cuerpo para la cual se desean encontrar ejercicios similares.
        top_n (int): Número máximo de ejercicios similares a devolver.

    Returns:
        List: Lista de tuplas (título, descripción, similitud, parte del cuerpo) de los ejercicios similares encontrados.
    """
    # Filtrar ejercicios por la parte del cuerpo
    buscar_ejercicios = [exercise for exercise in exercise_data if exercise["BodyPart"].lower() == body_part.lower()]
    if not buscar_ejercicios:
        print(f"No se encontraron ejercicios para la parte del cuerpo '{body_part}' en el conjunto de datos.")
        return []

    # Preprocesar las descripciones de los ejercicios
    descripcion_ejercicio = [exercise["Desc"] for exercise in buscar_ejercicios]

    # Obtener embeddings de las descripciones de los ejercicios
    exercise_embeddings = model.encode(descripcion_ejercicio, convert_to_tensor=True)
    
    def cosine_similarity(query_embedding):
        """Calcula la similitud coseno entre el embedding de consulta y los embeddings de los ejercicios."""
        similarities = util.pytorch_cos_sim(query_embedding, exercise_embeddings)[0]
        similar_indices = similarities.argsort(descending=True)
        similar_exercises = []
        for i in similar_indices:
            title = buscar_ejercicios[i]["Title"]
            description = buscar_ejercicios[i]["Desc"]
            body_part = buscar_ejercicios[i]['BodyPart']
            similarity = similarities[i].item()
            # Agregar ejercicios similares a la lista
            if title not in [ex[0] for ex in similar_exercises]:
                similar_exercises.append((title, description, similarity, body_part))
            # Detener el bucle si se alcanza el número máximo de ejercicios similares
            if len(similar_exercises) >= top_n:
                break
        return similar_exercises

    return cosine_similarity

def find_description_by_title(title):
    """Busca la descripción de un ejercicio por su título."""
    for exercise in exercise_data:
        if exercise["Title"].lower() == title.lower():
            return exercise["Desc"]
    return None

def find_similar_exercises_by_title(title, n=5):
    """
    Utilización del modelo: Encuentra ejercicios similares para un título de ejercicio dado.
    
    Args:
        title (str): Título del ejercicio para el cual se desean encontrar ejercicios similares.
        n (int): Número máximo de ejercicios similares a devolver.

    Returns:
        List: Lista de tuplas (título, descripción, parte del cuerpo, similitud) de los ejercicios similares encontrados.
    """
    query_description = find_description_by_title(title)
    if query_description is None:
        print("El ejercicio no fue encontrado.")
        return []

    # Preprocesar las descripciones de los ejercicios
    descriptions = [query_description] + [exercise["Desc"] for exercise in exercise_data if exercise["Title"].lower() != title.lower()]
    body_parts = [exercise["BodyPart"] for exercise in exercise_data if exercise["Title"].lower() != title.lower()]
    embeddings = model.encode(descriptions)

    query_embedding = embeddings[0].reshape(1, -1)
    description_embeddings = embeddings[1:]

    #Calcular similitud coseno entre el embedding de consulta y los embeddings de las descripciones de los ejercicios
    similarities = cosine_similarity(query_embedding, description_embeddings)[0]

    similar_indices = similarities.argsort()[::-1][:n]
    similar_exercises = [(exercise_data[i+1]["Title"], descriptions[i], body_parts[i], similarities[i]) for i in similar_indices]

    return similar_exercises


#Cargar el modelo y el tokenizador entrenados
model_path = "../lab/chatbot/modelo/gpt2-question-answering"
model_chatbot = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ajustar el token de padding

# Ajustar el tokenizer para evitar advertencias
tokenizer.padding_side = "left"

#Función para generar respuestas en formato HTML
def generate_response(question, model_chatbot, tokenizer, max_length=512, max_new_tokens=50):
        """
     Genera una respuesta basada en la pregunta proporcionada utilizando un modelo GPT-2.

     Args:
         question (str): La pregunta para la cual se desea generar una respuesta.
         model_chatbot: El modelo GPT-2 para generación de lenguaje.
         tokenizer: El tokenizador asociado al modelo.
         max_length (int, optional): La longitud máxima de la secuencia de entrada. Defaults to 512.
         max_new_tokens (int, optional): El número máximo de tokens nuevos en la respuesta generada. Defaults to 50.

     Returns:
         str: La respuesta generada para la pregunta dada.
     """
        # Preparar el prompt
        prompt = (
            "You are an expert fitness assistant known for providing clear and concise answers to users' questions. "
            f"Question: {question}"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
        # Generar la respuesta
        output = model_chatbot.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id  # Asegurar el uso del token de padding correcto
        )
        # Decodificar la respuesta
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        # Extraer solo la parte de la respuesta generada
        answer = answer.replace(prompt, "").strip()
        return answer

