import streamlit as st
import pandas as pd
import pickle
import os
from PIL import Image

# --- Custom Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #f0f2f6;
        color: #333;
    }
    .stApp {
        max-width: 800px !important;
        margin: 0 auto;
        padding: 2rem;
    }
    .st-container {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .logo-img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    .st-header h1 {
        color: #007bff;
        text-align: center;
    }
    .st-subheader {
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .st-selectbox label, .st-slider label, .st-number-input label, .st-radio label {
        color: #555;
        font-weight: bold;
    }
    .st-selectbox div > div > div > div, .st-slider div > div > div, .st-number-input div > div > input, .st-radio div > label {
        border-color: #ccc;
        border-radius: 5px;
    }
    .st-button > button {
        background-color: #28a745; /* Green button */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        border: none;
        cursor: pointer;
    }
    .st-button > button:hover {
        background-color: #218838; /* Darker green on hover */
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        text-align: center;
    }
    .average-spend {
        color: #28a745;
    }
    .model-info {
        margin-top: 1rem;
        font-style: italic;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load the Model ---
try:
    with open('modelo-reg-tree-knn-nn.pkl', 'rb') as file:
        model_Tree, model_Knn, model_NN, variables, min_max_scaler = pickle.load(file)
except FileNotFoundError:
    st.error("El archivo del modelo 'modelo-reg-tree-knn-nn.pkl' no se encontró. Asegúrate de que esté en la misma carpeta que este script.")
    st.stop()
except Exception as e:
    st.error(f"Ocurrió un error al cargar el modelo: {e}")
    st.stop()

# --- Main Content ---
st.container()

# --- Logo and Title ---
st.image("juegos.jpg", caption="Predicción de Gasto en Videojuegos", width=150, use_container_width=True)
st.header("Predicción de Gasto en Videojuegos")
st.subheader("Ingresa tus preferencias para obtener una estimación de gasto")

# --- Input Fields ---
edad = st.slider("Edad:", min_value=14, max_value=120, value=30, step=1)
genero = st.selectbox("Género:", ["Hombre", "Mujer", "Otro"])
tipo_juego = st.selectbox(
    "Tipo de Videojuego:",
    ["Mass Effect", "Sim City", "Dead Space", "Battlefield", "FIFA", "F1", "KOA: Reckoning", "Crysis"]
)
plataforma = st.selectbox(
    "Plataforma:",
    ["PC", "Xbox", "Play Station", "Otros"]
)
consumidor_habitual = st.selectbox("¿Eres consumidor habitual?", ["Si", "No"]) # Changed to selectbox

# --- Button for Prediction ---
if st.button("Realizar Predicción"):
    # Crea un DataFrame con los datos de entrada del usuario
    input_data = pd.DataFrame({
        'Edad': [edad],
        'videojuego': [tipo_juego],
        'Plataforma': [plataforma],
        'Sexo': [genero],
        'Consumidor_habitual': [consumidor_habitual]
    })

    # Muestra los datos ingresados en una tabla
    st.subheader("Datos Ingresados:")
    st.table(input_data)

    # **Preprocesamiento de datos para el modelo**
    data_preparada = input_data.copy()
    data_preparada = pd.get_dummies(data_preparada, columns=['videojuego'], prefix='videojuego')
    data_preparada = pd.get_dummies(data_preparada, columns=['Plataforma'], prefix='Plataforma')
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], prefix='Sexo')
    data_preparada = pd.get_dummies(data_preparada, columns=['Consumidor_habitual'], prefix='Consumidor_habitual')
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    data_preparada[['Edad']] = min_max_scaler.transform(data_preparada[['Edad']])

    try:
        # Realiza la predicción con los tres modelos
        prediccion_tree = model_Tree.predict(data_preparada.values)[0]
        prediccion_knn = model_Knn.predict(data_preparada.values)[0]
        prediccion_nn = model_NN.predict(data_preparada.values)[0]

        # Calcula el promedio de las predicciones
        prediccion_promedio = (prediccion_tree + prediccion_knn + prediccion_nn) / 3

        # Muestra el resultado de la predicción promediada
        st.subheader("Predicción de Gasto")
        st.markdown(f"<p class='prediction-result average-spend'>Se estima que gastarás: ${prediccion_promedio:.2f} en este escenario.</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='model-info'>Modelos utilizados: Árbol de Decisión, KNN, Red Neuronal</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")