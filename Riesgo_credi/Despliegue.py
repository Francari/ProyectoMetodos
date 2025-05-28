import streamlit as st
import joblib
import pandas as pd

# Cargar modelo con caché para mejorar rendimiento
@st.cache_resource # Usar st.cache_data si st.cache_resource no está disponible en tu versión de Streamlit
def cargar_modelo():
    modelo = joblib.load("modelo_crediticio.pkl")
    return modelo

modelo = cargar_modelo()

# Formulario de ingreso
st.title("Evaluación de Riesgo Crediticio")
st.write("Por favor, ingresa los siguientes datos para evaluar el riesgo de incumplimiento.Ten presente que los valores deben expresarse en dólares")

# Campos del formulario basados en tu dataset
Edad = st.slider("Edad", min_value=18, max_value=90, value=30)
Ingreso_Mensual = st.number_input("Ingreso Mensual", min_value=0.0, value=5000.0, step=500.0)
Historial_Crediticio = st.slider("Historial Crediticio (300-850)", min_value=0, max_value=850, value=700)
Deuda_Actual = st.number_input("Deuda Actual", min_value=0.0, value=1500.0, step=500.0)

# Opciones para campos categóricos
tipo_empleo_opciones = ['Desempleado', 'Empleado', 'Estudiante', 'Independiente'] # Asegúrate de que estas opciones coincidan con las categorías usadas en el entrenamiento de tu modelo
Tipo_Empleo = st.selectbox("Tipo de Empleo", tipo_empleo_opciones)

Tiempo_Empleo = st.slider("Tiempo en Empleo (Meses)", min_value=0, max_value=240, value=24)
Numero_Productos = st.slider("Número de Productos Crediticios", min_value=0, max_value=10, value=2)

estado_civil_opciones = ['Viudo', 'Soltero', 'Casado', 'Divorciado'] # Asegúrate de que estas opciones coincidan
Estado_Civil = st.selectbox("Estado Civil", estado_civil_opciones)

sexo_opciones = ['Masculino', 'Femenino'] # Asegúrate de que estas opciones coincidan
Sexo = st.selectbox("Sexo", sexo_opciones)

Monto_Prestamo = st.number_input("Monto del Préstamo", min_value=0.0, value=15000.0, step=1000.0)
Tiempo_Meses_Prestamo = st.slider("Tiempo del Préstamo (meses)", min_value=1, max_value=120, value=36)


# Botón de predicción
if st.button("Evaluar riesgo"):
    # Crear un diccionario con los datos del usuario
    # ¡IMPORTANTE! Los nombres de las claves (columnas) deben coincidir EXACTAMENTE
    # con los nombres de las columnas que el modelo espera (usados durante el entrenamiento).
    # Verifica especialmente el nombre para 'Numero de Productos Crediticios'.
    datos_entrada_dict = {
        'Edad': Edad,
        'Ingreso_Mensual': Ingreso_Mensual,
        'Historial_Crediticio': Historial_Crediticio,
        'Deuda_Actual': Deuda_Actual,
        'Tipo_Empleo': Tipo_Empleo,
        'Tiempo_Empleo': Tiempo_Empleo,
        'Numero_Productos': Numero_Productos,
        'Estado_Civil': Estado_Civil,
        'Sexo': Sexo,
        'Monto_Prestamo': Monto_Prestamo,
        'Tiempo_Meses_Prestamo': Tiempo_Meses_Prestamo
    }
    # Convertir el diccionario a un DataFrame de Pandas
    # Cada valor debe estar en una lista para crear una fila.
    datos_usuario = pd.DataFrame({k: [v] for k, v in datos_entrada_dict.items()})
    pred = modelo.predict(datos_usuario)
    st.write("❗ Riesgo de incumplimiento" if pred[0]==1 else "✅ Bajo riesgo")
