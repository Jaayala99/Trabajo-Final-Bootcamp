import cherrypy
import os
from proceso_inferencia import find_similar_exercises_by_title, find_similar_exercises, model, clean_description, generate_response, model_chatbot,tokenizer
import random

class ExerciseApp:
    @cherrypy.expose
    def index(self):
        return open("templates/index.html")

    @cherrypy.expose
    def buscar_similitud(self, sustituir=None):
        try:
            if sustituir is None or sustituir.strip() == "":
                return "Por favor, proporcione un título de ejercicio válido para buscar similitudes."
            
            exercise_title = sustituir
            similar_exercises = find_similar_exercises_by_title(exercise_title)

            if not similar_exercises:
                return "No se encontraron ejercicios similares para el título proporcionado."

            # Crear una cadena HTML con los ejercicios similares
            similar_exercises_html = "<ul>"
            for title, description, body_part, similarity in similar_exercises:
                similar_exercises_html += f"<li><strong>Ejercicio:</strong> {title} (Similitud: {similarity:.2f}%)<br><strong>Descripción:</strong> {description}<br><strong>Parte del cuerpo:</strong> {body_part}</li><br>"
            similar_exercises_html += "</ul>"
            return similar_exercises_html
        except FileNotFoundError:
            return "Archivo de ejercicios no encontrado."
        except Exception as e:
            return f"Error al buscar ejercicios similares: {str(e)}"

    
    @cherrypy.expose
    def respuesta_ejercicio(self, parte_cuerpo=None, parte_cuerpo_nuevo=None):
        try:
            parte_cuerpo = parte_cuerpo or parte_cuerpo_nuevo  # Toma uno de los parámetros si está presente
            if not parte_cuerpo:
                return "Debe proporcionar una parte del cuerpo para buscar ejercicios similares."
            
            devolver_ejercicios_similares = find_similar_exercises(parte_cuerpo)
            if devolver_ejercicios_similares:
                ejercicios_similares = devolver_ejercicios_similares(model.encode(clean_description("")))
                random_ejercicio = random.choice(ejercicios_similares)
                title, description, similarity, body_part = random_ejercicio
                resp = (f"<strong>Ejercicio:</strong> {title}<br><strong>Descripción:</strong> {description}<br><strong>Parte del cuerpo:</strong> {body_part}<br><br>")
                return resp
            else:
                return "No se encontraron ejercicios para esa parte del cuerpo."
        except Exception as e:
            return f"Error al obtener respuesta del ejercicio: {str(e)}"
        

    @cherrypy.expose
    def generar_respuesta(self, chatbot_input=None):
        try:
            question = chatbot_input
            response = generate_response(question, model_chatbot, tokenizer)
            response = response.replace("Answer: ", "").strip()
            return response  # Devuelve la respuesta generada al cliente
        except Exception as e:
            return f"Error al generar respuesta: {str(e)}"
       
        



if __name__ == "__main__":
    conf = {
        "/": {
            "tools.staticdir.on": True,
            "tools.staticdir.dir": os.path.abspath("app/templates")
        }
    }
    cherrypy.quickstart(ExerciseApp(), "/", conf)
