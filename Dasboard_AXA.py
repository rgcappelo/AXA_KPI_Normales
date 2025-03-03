import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Page configuration
st.set_page_config(
    page_title="Análisis Predictivo de Riesgos Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 22px;
        font-weight: bold;
        color: #2563EB;
        margin-top: 30px;
    }
    .metric-container {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #2563EB;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title and description
st.markdown('<div class="main-header">Dashboard de Análisis Predictivo de Riesgos</div>', unsafe_allow_html=True)

# Agregar la información del OKR
with st.expander("Objetivo y Key Results (OKR)", expanded=False):
    st.markdown("""
    **Objetivo:** Implementar un sistema de análisis predictivo para riesgos emergentes en los próximos 12 meses.
    
    **Key Results (KR):**
    - KR1: Desarrollar 5 modelos predictivos de riesgo basados en datos de clientes y mercados en 6 meses.
    - KR2: Aumentar la precisión de predicción de riesgos en un 20% en 12 meses.
    - KR3: Integrar el 100% de los datos históricos relevantes en el sistema de análisis predictivo.
    """)

# Crear datos de fecha
fechas = pd.date_range(start='2022-01-01', end='2025-07-31', freq='MS').strftime('%Y-%m')

# Data preparation
precision_modelo = [72.4, 69.3, 73.2, 77.6, 68.8, 70.5, 72.1, 74.3, 71.8, 75.6, 
                    78.2, 76.4, 79.1, 77.8, 80.3, 81.5, 78.4, 82.2, 79.9, 84.3, 
                    81.5, 85.1, 83.2, 86.4, 85.6, 84.2, 87.3, 86.1, 88.2, 87.9, 
                    89.5, 88.3, 90.1, 89.7, 91.2, 90.5, 91.8, 92.0, 93.2, 93.5, 
                    94.1, 94.8, 95.3]

alertas_generadas = [10, 12, 15, 8, 17, 19, 14, 11, 13, 18, 16, 9, 14, 15, 21, 13, 20, 23, 12, 18, 
                     17, 22, 16, 19, 24, 18, 25, 20, 22, 19, 23, 21, 24, 22, 26, 23, 25, 27, 29, 28, 
                     30, 31, 32]

dias_anticipacion = [12, 14, 11, 9, 16, 18, 13, 10, 15, 17, 14, 12, 19, 16, 21, 20, 18, 22, 17, 24, 
                     21, 25, 22, 26, 23, 27, 24, 28, 25, 30, 26, 29, 27, 30, 28, 31, 29, 32, 33, 34, 
                     35, 36, 37]

# Create DataFrame
df = pd.DataFrame({
    'Fecha': fechas,
    'Precisión del Modelo (%)': precision_modelo,
    'Número de Alertas': alertas_generadas,
    'Días de Anticipación': dias_anticipacion
})

# Convertir fechas a datetime para operaciones
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Determinar límite de datos históricos vs. pronósticos (Febrero 2025)
fecha_corte = pd.to_datetime('2025-02-01')
df['Es Proyección'] = df['Fecha'] >= fecha_corte

# Métricas resumidas
col1, col2, col3 = st.columns(3)

# Calcular valores actuales (últimos datos históricos)
df_hist = df[~df['Es Proyección']]
df_proj = df[df['Es Proyección']]

precision_actual = df_hist['Precisión del Modelo (%)'].iloc[-1]
precision_inicial = df_hist['Precisión del Modelo (%)'].iloc[0]
mejora_precision = ((precision_actual - precision_inicial) / precision_inicial) * 100

alertas_actual = df_hist['Número de Alertas'].iloc[-1]
alertas_promedio = df_hist['Número de Alertas'].mean()
cambio_alertas = ((alertas_actual - alertas_promedio) / alertas_promedio) * 100

dias_actual = df_hist['Días de Anticipación'].iloc[-1]
dias_inicial = df_hist['Días de Anticipación'].iloc[0]
mejora_dias = ((dias_actual - dias_inicial) / dias_inicial) * 100

# Mostrar KPIs
with col1:
    st.metric(
        label="Precisión Actual del Modelo",
        value=f"{precision_actual:.1f}%",
        delta=f"{mejora_precision:.1f}% desde inicio"
    )

with col2:
    st.metric(
        label="Alertas Generadas (Último Mes)",
        value=f"{alertas_actual}",
        delta=f"{cambio_alertas:.1f}% vs promedio"
    )

with col3:
    st.metric(
        label="Días de Anticipación",
        value=f"{dias_actual}",
        delta=f"{mejora_dias:.1f}% desde inicio"
    )

# Tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Precisión del Modelo", "Alertas Generadas", "Días de Anticipación"])

# Tab 1: Precisión del Modelo
with tab1:
    st.markdown('<div class="sub-header">Evolución de la Precisión del Modelo Predictivo</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="info-box">Este indicador muestra la precisión de los modelos predictivos de riesgo a lo largo del tiempo. La línea roja discontinua marca el inicio de las proyecciones en febrero de 2025.</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=df_hist['Fecha'],
            y=df_hist['Precisión del Modelo (%)'],
            mode='lines+markers',
            name='Datos Históricos',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8, color='#2563EB')
        ))
        
        # Datos proyectados
        fig.add_trace(go.Scatter(
            x=df_proj['Fecha'],
            y=df_proj['Precisión del Modelo (%)'],
            mode='lines+markers',
            name='Proyecciones',
            line=dict(color='#818CF8', width=3, dash='dot'),
            marker=dict(size=8, color='#818CF8')
        ))
        
        # Línea vertical que marca la fecha de corte
        fig.add_shape(
            type="line",
            x0=fecha_corte,
            y0=60,
            x1=fecha_corte,
            y1=100,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Meta del OKR (20% de mejora desde el inicio)
        meta_precision = precision_inicial * 1.2
        fig.add_shape(
            type="line",
            x0=df['Fecha'].min(),
            y0=meta_precision,
            x1=df['Fecha'].max(),
            y1=meta_precision,
            line=dict(color="green", width=2, dash="dash"),
        )
        
        # Anotación para la meta
        fig.add_annotation(
            x=df['Fecha'].max(),
            y=meta_precision,
            text="Meta OKR (+20%)",
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=-30,
            font=dict(color="green", size=12),
        )
        
        # Actualizar el diseño
        fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Precisión (%)',
            yaxis_range=[60, 100],
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Alertas Generadas
with tab2:
    st.markdown('<div class="sub-header">Evolución de Alertas Generadas por el Sistema</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="info-box">Este indicador refleja la capacidad del modelo para identificar riesgos en tiempo real. Un aumento en las alertas puede significar una mayor sensibilidad del modelo.</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Bar(
            x=df_hist['Fecha'],
            y=df_hist['Número de Alertas'],
            name='Datos Históricos',
            marker_color='#2563EB'
        ))
        
        # Datos proyectados
        fig.add_trace(go.Bar(
            x=df_proj['Fecha'],
            y=df_proj['Número de Alertas'],
            name='Proyecciones',
            marker_color='#818CF8',
            marker_pattern_shape="/"
        ))
        
        # Línea vertical que marca la fecha de corte
        fig.add_shape(
            type="line",
            x0=fecha_corte,
            y0=0,
            x1=fecha_corte,
            y1=35,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Línea promedio
        fig.add_trace(go.Scatter(
            x=df['Fecha'],
            y=[df_hist['Número de Alertas'].mean()] * len(df),
            mode='lines',
            name='Promedio Histórico',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Actualizar el diseño
        fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Número de Alertas',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Días de Anticipación
with tab3:
    st.markdown('<div class="sub-header">Evolución del Tiempo Medio de Anticipación en la Detección de Riesgos</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="info-box">Este indicador muestra la cantidad promedio de días que el modelo predice un evento de riesgo antes de que ocurra. Un mayor número de días permite tomar mejores decisiones estratégicas.</div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=df_hist['Fecha'],
            y=df_hist['Días de Anticipación'],
            mode='lines+markers',
            name='Datos Históricos',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8, color='#2563EB')
        ))
        
        # Datos proyectados
        fig.add_trace(go.Scatter(
            x=df_proj['Fecha'],
            y=df_proj['Días de Anticipación'],
            mode='lines+markers',
            name='Proyecciones',
            line=dict(color='#818CF8', width=3, dash='dot'),
            marker=dict(size=8, color='#818CF8')
        ))
        
        # Línea vertical que marca la fecha de corte
        fig.add_shape(
            type="line",
            x0=fecha_corte,
            y0=0,
            x1=fecha_corte,
            y1=40,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        # Actualizar el diseño
        fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Días de Anticipación',
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Análisis y recomendaciones
st.markdown('<div class="sub-header">Análisis y Recomendaciones</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Análisis de Tendencias
        
        - **Precisión del Modelo**: Ha mejorado significativamente desde el inicio, pasando de 72.4% a 90.5% actualmente, superando el objetivo del OKR de aumentar la precisión en un 20%.
        
        - **Alertas Generadas**: Se observa un aumento consistente en el número de alertas mensuales, lo que sugiere una mayor capacidad de detección del sistema.
        
        - **Días de Anticipación**: El tiempo de anticipación ha aumentado de 12 días a aproximadamente 30 días, lo que proporciona un margen de maniobra considerablemente mayor para la gestión de riesgos.
        """)
    
    with col2:
        st.markdown("""
        ### Recomendaciones
        
        1. ✅ **Continuar mejorando los modelos**: Aunque ya se ha superado el objetivo de precisión, se recomienda seguir refinando los algoritmos para mantener el rendimiento frente a nuevos tipos de riesgos.
        
        2. ✅ **Optimizar infraestructura**: Implementar procesamiento en tiempo real para reducir aún más el tiempo de detección y respuesta.
        
        3. ✅ **Análisis trimestral**: Establecer revisiones trimestrales de sensibilidad del modelo para ajustar parámetros según las condiciones cambiantes del mercado.
        """)

# Filtros interactivos
with st.sidebar:
    st.header("Filtros y Controles")
    
    # Selector de rango de fechas
    st.subheader("Rango de Fechas")
    start_date = st.date_input(
        "Fecha de inicio",
        pd.to_datetime('2022-01-01').date(),
        min_value=pd.to_datetime('2022-01-01').date(),
        max_value=pd.to_datetime('2025-07-31').date()
    )
    
    end_date = st.date_input(
        "Fecha de fin",
        pd.to_datetime('2025-07-31').date(),
        min_value=pd.to_datetime('2022-01-01').date(),
        max_value=pd.to_datetime('2025-07-31').date()
    )
    
    # Selector de vista (histórica, proyecciones, ambas)
    st.subheader("Tipo de Datos")
    vista_seleccionada = st.radio(
        "Mostrar:",
        ["Todos los datos", "Solo históricos", "Solo proyecciones"]
    )
    
    # Casilla para mostrar/ocultar líneas de meta
    mostrar_metas = st.checkbox("Mostrar líneas de meta", value=True)
    
    # Botón para descargar datos
    st.subheader("Exportar Datos")
    if st.download_button(
        label="Descargar CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="datos_predictivos_riesgos.csv",
        mime="text/csv"
    ):
        st.success("Datos descargados correctamente")

# Nota: En una aplicación real, estos filtros se conectarían a los gráficos para filtrar 
# los datos según las selecciones del usuario
