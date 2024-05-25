# Autor
Juan Antonio Ayala García


# Generador de Rutinas
Este proyecto es un buscador de ejercicios físicos que permite a los usuarios encontrar ejercicios para entrenar diferentes partes del cuerpo. Los usuarios pueden ingresar la parte del cuerpo que desean entrenar y obtener un ejercicio relacionado. Además, el proyecto permite a los usuarios agregar múltiples ejercicios a su rutina de entrenamiento, buscar ejercicios similares y calificar su experiencia.


# Características
- **Búsqueda de Ejercicios**: Los usuarios pueden buscar ejercicios específicos ingresando la parte del cuerpo que desean entrenar.
- **Visualización de Resultados**: Los resultados de la búsqueda se muestran de manera clara, incluyendo el título del ejercicio, su descripción y la parte del cuerpo asociada.
- **Agregar Ejercicios a la Rutina**: Los usuarios pueden agregar ejercicios a su rutina de entrenamiento con un solo clic.
- **Buscar ejercicios Similares**: Los usuarios pueden buscar ejercicios similares a un ejercicio proporcionado con un solo clic.
- **Calificación de la Rutina**: Al finalizar su rutina, los usuarios pueden calificar su experiencia de entrenamiento mediante un sistema de estrellas.


# Tecnologías Utilizadas
- **Lenguajes de Programación**
    - JavaScript
    - Python (incluyendo bibliotecas como Pandas, Numpy, Transformers, Sickit-Learn, Matplotlib)
- **Hojas de Estilo**
    - CSS
- **Frameworks**
    - Boostrap
    - CherryPy


# Uso
1. **Búsqueda de Ejercicios**: Ingresa la parte del cuerpo que deseas entrenar (Chest , Biceps , Triceps...) en el campo de búsqueda y presiona "Buscar".
2. **Agregar Ejercicios**: Después de obtener los resultados de la búsqueda, puedes agregar ejercicios a tu rutina haciendo clic en el botón "Añadir ejercicio".
3. **Limpiar Rutina**: Una vez que hayas terminado de agregar ejercicios, puedes finalizar tu rutina haciendo clic en el botón "Finalizar rutina".
4. **Calificar App**: Después de finalizar la rutina, puedes calificar tu experiencia de entrenamiento haciendo clic en las estrellas correspondientes.
5. **Buscar Similares**: Tenemos la opción de buscar ejercicios similares introduciendo el título del ejercicio.


# Aplicación de Inteligencia Artificial en Generación de Rutinas de Ejercicio

En este proyecto, hemos desarrollado una aplicación de inteligencia artificial (IA) que se centra en la generación de rutinas de ejercicio personalizadas. A continuación, se presentan las principales tareas realizadas durante el desarrollo:

**Análisis de Datos**

- **Exploración de Datasets**: Se llevaron a cabo análisis exploratorios en diferentes conjuntos de datos relacionados con ejercicios físicos para comprender la estructura y la calidad de los datos.

- **Limpieza de Datos**: Se aplicaron técnicas de limpieza de datos para abordar valores faltantes, valores atípicos y datos inconsistentes que podrían afectar la calidad de los modelos posteriores.

**Preprocesamiento de Datos**

- **Codificación de Variables Categóricas**: Se convirtieron las variables categóricas en variables numéricas utilizando técnicas como one-hot encoding o label encoding, según la naturaleza de los datos.

- **Embeddings para Textos**: Se utilizaron técnicas de embeddings de texto, como el modelo BERT, para representar descripciones de ejercicios en forma de vectores numéricos densos, que luego se utilizaron como características en los modelos de generación de rutinas.


**Desarrollo de Modelos**

- **Funciones Principales del Generador de Rutinas**: Se desarrollaron funciones principales para generar rutinas de ejercicio personalizadas. Estas funciones utilizan modelos como paraphrase-MiniLM-L6-v2 y search_similarity para encontrar ejercicios similares o relacionados.

- **Modelos Preentrenados**: Se integraron modelos preentrenados, como BERT, para generar embeddings de texto de alta calidad, que se utilizaron en la búsqueda y recomendación de ejercicios.


# Arquitectura del proyecto 
```
TRABAJO/
├── inference/
│ └── inferencie.py
├── lab/
│ ├── analisis_dataset/
│ │ ├── analisis.ipynb
│ │ └── preprocess/
│ ├── dataset/
│ │ ├── megaGymDataset.csv
│ │ ├── megaGymDataset_clean.csv
│ │ ├── megaGymDataset_balanceado.csv
│ │ └── megaGymDataset_predecir.csv
│ ├── ipynb/
│ │ ├── funciones_principales/
│ │ │ ├── generar_ejercicios.ipynb
│ │ │ └── similitudes_ejercicios.ipynb
│ │ └── modelo_regresion/
│ │ ├── preprocess_predecir_musculos.ipynb
│ │ └── train_predecir_musculos.ipynb
│ └── json_data/
│ ├── datos_ejercicios.json
│ ├── datos_ejercicios_submuestreo.json
│ └── convert_json.ipynb
└── proyecto/
├── datos_json/
│ └── datos_ejercicios_submuestreo.json
├── script/
│ ├── parte_cuerpo.py
│ ├── predecir_musculo.py
│ └── similar_by_titulo.py
├── templates/
│ └── index.html
└── app.py
└── proceso_inferencia.py
```

**INFERENCE/**
- inference.py ----> Archivo donde reorganizamos todo nuestro flujo de trabajo el cual nos ayuda a tener el codigo de nuestro proyecto de manera más clara y a la gora de solucionar problemas es más entendible y fácil dar con un posible error

**LAB/**
- **analisis_dataset/** 

    - analisis.ipynb ----> Archivo donde analizamos todas las variables de nuestro dataset, valores NaN, posibles outliers, explicación de nuestras variables...
    - preprocess_dataset.ipynb ----> Archivo donde preprocesamos todo nuestro dataset, nos encargamos de la normalizacion de nuestras variables, clasificación de las variables categoricas...

- **dataset/** 
    - megaGymDataset.csv 
    - megaGymDtaset_clean.csv 
    - megaGymDtaset_balanceado.csv 
    - megaGymDtaset_predecir.csv 

- **ipynb/** 
    - **funciones_principales/**
        - generar_ejercicios.ipynb ----> Archivo que contiene una de las dos funciones principales que se encarga de generar ejercicios según la parte del cuerpo
        - similitudes_ejercicios.ipynb ----> Archivo que se encarga de buscar similitudes entre los ejercicios
    - **modelo_regresion/**
        - preprocess_predecir_musculos.ipynb ----> Archivo donde preprocesamos los datos de nuestro csv para poder entrenar nuestro modelo
        - train_predecir_musculos.ipynb ----> Archivo donde entrenamos nuestro modelo de regresión oara predecir que parte del cuerpo es la involucrada en un ejercicio dado 

- **json_data/** 
    - convert_json.ipynb
    - datos_ejercicios_sybmuestreo.json
    - datos_ejercicios.json


**PROYECTO/**

- **datos_json/**
    - datos_ejercicios_sybmuestreo.json

- **script/**
    - parte_cuerpo.py ----> Archivo donde se encuentra nuestra función que genera al usuario ejercicios según una parte del cuerpo dada
    - predecir_musculo.py ----> Archivo .py donde diseñamos la funcion encargada de predecir el musculo a entrenar dada una descripción
    - similar_by_tittle.py ----> Archivo .py similar a similitudes_ejercicios.ipynb donde cargamos nuestro modelo preentrenado y realizamos nuestra funcion de search_similarity

- **templates/**
    - index.html ----> Archivo html donde "diseñamos" nuestra web con ayuda de JavaScript

- app.py ----> Archivo donde se encuentra alojado nuestro servidor local usando cherrypy y se encarga de renderizar nuestro index.html e implementar nuestras funciones
- proceso_inferencia.py ----> Archivo donde se reorganiza todo nuestro flujo de trabajo y se encarga de importar las funciones principales apuntando a nuestro servidor

enviroment.yml ----> Archivo donde se almacenan todas las dependencias de nuestro entorno y su información

# Archivos utilizados 
- **Dataset Kaggle** = https://www.kaggle.com/datasets/niharika41298/gym-exercise-data
- **Modelo HuggingFace** = https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2