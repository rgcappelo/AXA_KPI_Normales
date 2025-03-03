import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Cargar el contenido del documento Word en la introducción
intro_text = """
## CONSTRUCCIÓN DEL NUEVO KPI: NET PROMOTER SCORE (NPS) AVANZADO PARA AXA

### Introducción

Basándonos en el análisis previo del caso de AXA y en las recomendaciones consensuadas en el panel de expertos, desarrollaremos un nuevo KPI de Net Promoter Score (NPS) que supere las limitaciones tradicionales y se integre con los KPIs normales de AXA.

Este nuevo KPI responderá a tres perspectivas clave:

- **NPS Predictivo y Accionable (dNPS):** Enfocado en la anticipación en tiempo real mediante machine learning y datos de comportamiento.
- **NPS de Confianza y Viralidad (tNPS + sNPS):** Reflejando la confianza en la aseguradora y su capacidad de generar recomendaciones orgánicas en redes sociales.
- **NPS Personalizado por Sector:** Adaptando las métricas a seguros, banca o negocios digitales, según la naturaleza del cliente.

### Evaluación del Modelo Holt-Winters

| Variable   | MSE   | RMSE  | R²   |
|------------|--------|--------|------|
| **dNPS** (Dynamic NPS) | 262.92 | 16.21 | -2.57 |
| **tNPS** (Trust-Based NPS) | 64.93  | 8.05  | -0.17 |
| **sNPS** (Social NPS) | 155.60 | 12.47 | -0.30 |
| **Smart_NPS** (Combinado) | 55.60  | 7.45  | -0.54 |

🔹 **Interpretación de los Resultados:**
- El modelo Holt-Winters predice bien el **tNPS y Smart_NPS**, pero tiene menor precisión para **dNPS y sNPS** (posiblemente por variabilidad en los datos sociales y en tiempo real).
- El **R² negativo** indica que Holt-Winters no es el modelo óptimo para ciertas variables, por lo que podría mejorarse con SARIMA o Machine Learning para predecir tendencias más complejas.

### Descripción de los Datos Necesarios y Cómo Obtenerlos

| **Variable** | **Fuente de Datos** | **Método de Obtención** | **Transformaciones Necesarias** |
|-------------|----------------------|------------------------|--------------------------------|
| **dNPS** (Dynamic NPS) | Datos transaccionales, interacciones con atención al cliente, reclamos | Modelos de predicción con Machine Learning y análisis de tendencias | Normalización de datos, detección de anomalías |
| **tNPS** (Trust-Based NPS) | Encuestas de satisfacción de largo plazo, tasas de retención | Cálculo de confianza basado en series temporales y satisfacción post-reclamación | Análisis de correlación con Net Retention Rate (NRR) |
| **sNPS** (Social NPS) | Redes sociales, menciones en foros y reviews online | Análisis de sentimiento con NLP, tracking de menciones de marca | Clasificación de menciones en positivas, neutras y negativas |
| **Smart_NPS** (NPS Combinado) | Integración de dNPS, tNPS y sNPS | Fórmula ponderada personalizada | Ajuste de pesos según la industria |
"""

# Generación de datos simulados
np.random.seed(42)
dates = pd.date_range(start="2022-01-01", periods=37, freq='M')
dNPS = np.clip(np.random.normal(loc=60, scale=10, size=len(dates)), 40, 90)
tNPS = np.clip(np.random.normal(loc=70, scale=8, size=len(dates)), 50, 95)
sNPS = np.clip(np.random.normal(loc=55, scale=12, size=len(dates)), 30, 85)
smart_NPS = (0.5 * dNPS) + (0.3 * tNPS) + (0.2 * sNPS)

df_nps = pd.DataFrame({
    'Fecha': dates,
    'dNPS': dNPS,
    'tNPS': tNPS,
    'sNPS': sNPS,
    'Smart_NPS': smart_NPS
})

# Streamlit Dashboard
st.set_page_config(page_title="Dashboard Smart NPS AXA", layout="wide")
st.title("Dashboard Smart NPS AXA")
st.markdown(intro_text)

# Generación de gráficos
st.header("Estamos logrando satisfacer las necesidades de sus clientes?")
fig1, ax1 = plt.subplots()
ax1.plot(df_nps['Fecha'], df_nps['dNPS'], marker='o', linestyle='-', label="dNPS")
ax1.set_xlabel("Fecha")
ax1.set_ylabel("Puntuación")
ax1.legend()
st.pyplot(fig1)

st.header("Somos capaces de mantener la confianza de los clientes en el largo plazo?")
fig2, ax2 = plt.subplots()
ax2.plot(df_nps['Fecha'], df_nps['tNPS'], marker='s', linestyle='-', color='orange', label="tNPS")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Puntuación")
ax2.legend()
st.pyplot(fig2)

st.header("Tenemos fortaleza y buena percepción de los clientes en redes sociales?")
fig3, ax3 = plt.subplots()
ax3.plot(df_nps['Fecha'], df_nps['sNPS'], marker='d', linestyle='-', color='green', label="sNPS")
ax3.set_xlabel("Fecha")
ax3.set_ylabel("Puntuación")
ax3.legend()
st.pyplot(fig3)

st.header("Logramos satisfacer las necesidades del cliente, manteniendo su confianza en el largo plazo y con fuerte percepción en redes?")
fig4, ax4 = plt.subplots()
ax4.plot(df_nps['Fecha'], df_nps['Smart_NPS'], marker='x', linestyle='-', color='blue', label="Smart NPS")
ax4.set_xlabel("Fecha")
ax4.set_ylabel("Puntuación")
ax4.legend()
st.pyplot(fig4)
