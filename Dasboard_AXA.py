import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis Predictivo de Riesgos",
    page_icon="📊",
    layout="wide"
)

# Título principal
st.title("Dashboard de Análisis Predictivo de Riesgos Emergentes")

# Crear datos de muestra
@st.cache_data
def generate_data():
    np.random.seed(42)  # Para reproducibilidad

    # Generación de fechas desde enero 2022 hasta julio 2025
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2025-07-31')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Generar datos de precisión con tendencia creciente pero realista
    precision_base = 65  # Comenzamos con 65% de precisión
    precision_values = []

    for i, date in enumerate(date_range):
        # Añadir tendencia creciente
        trend = min(20, i * 0.3)  # Incremento gradual hasta 20%
        
        # Añadir variabilidad mensual 
        seasonal = 2 * np.sin(i/12 * 2 * np.pi)
        
        # Añadir ruido aleatorio
        noise = np.random.normal(0, 1.5)
        
        # Combinar componentes 
        precision = precision_base + trend + seasonal + noise
        
        # Limitar al rango 60-90%
        precision = max(60, min(90, precision))
        
        precision_values.append(precision)

    # Crear DataFrame
    df = pd.DataFrame({
        'fecha': date_range,
        'precision_modelo': precision_values
    })

    # Añadir columna para indicar datos históricos vs proyecciones
    df['tipo'] = 'Histórico'
    df.loc[df['fecha'] > pd.to_datetime('2025-02-01'), 'tipo'] = 'Proyección'

    # Generar datos para Número de Alertas Generadas
    alertas_base = 80
    alertas_values = []

    for i, date in enumerate(date_range):
        trend = min(100, i * 1.5)  # Crecimiento según se mejora el modelo
        seasonal = 20 * np.sin(i/12 * 2 * np.pi)  # Estacionalidad anual
        noise = np.random.normal(0, 10)
        
        alertas = alertas_base + trend + seasonal + noise
        alertas = max(50, int(alertas))
        
        alertas_values.append(alertas)

    df['num_alertas'] = alertas_values

    # Generar datos para Días de Anticipación
    anticipacion_base = 10  # 10 días de anticipación inicial
    anticipacion_values = []

    for i, date in enumerate(date_range):
        trend = min(15, i * 0.2)  # Mejora gradual
        seasonal = 2 * np.sin(i/6 * 2 * np.pi)  # Ciclo semestral
        noise = np.random.normal(0, 1)
        
        anticipacion = anticipacion_base + trend + seasonal + noise
        anticipacion = max(5, min(30, anticipacion))
        
        anticipacion_values.append(anticipacion)

    df['dias_anticipacion'] = anticipacion_values
    
    return df, start_date, end_date

# Generar los datos
df, start_date, end_date = generate_data()

# Panel lateral con información del OKR y filtros
with st.sidebar:
    st.header("Objetivo del OKR")
    st.write("Implementar un sistema de análisis predictivo para riesgos emergentes en los próximos 12 meses.")
    
    st.subheader("Key Results (KR)")
    st.markdown("""
    - **KR1:** Desarrollar 5 modelos predictivos de riesgo basados en datos de clientes y mercados en 6 meses.
    - **KR2:** Aumentar la precisión de predicción de riesgos en un 20% en 12 meses.
    - **KR3:** Integrar el 100% de los datos históricos relevantes en el sistema de análisis predictivo.
    """)
    
    st.divider()
    
    # Filtros
    st.subheader("Filtros")
    
    # Selector de fechas
    date_range = st.date_input(
        "Rango de fechas",
        value=(start_date, end_date),
        min_value=start_date,
        max_value=end_date
    )
    
    if len(date_range) == 2:
        start_filter, end_filter = date_range
    else:
        start_filter, end_filter = start_date, end_date
    
    # Convertir a datetime para filtrar el dataframe
    start_filter = pd.to_datetime(start_filter)
    end_filter = pd.to_datetime(end_filter)
    
    # Selector de métricas
    selected_metrics = st.multiselect(
        "Seleccionar métricas",
        ["Precisión del Modelo", "Número de Alertas", "Días de Anticipación"],
        default=["Precisión del Modelo", "Número de Alertas", "Días de Anticipación"]
    )
    
    # Mostrar acciones necesarias
    st.divider()
    st.subheader("Acciones Necesarias")
    st.markdown("""
    - Entrenar y mejorar los modelos predictivos con datos más recientes y relevantes.
    - Optimizar el pipeline de datos para mejorar la integración de fuentes externas.
    - Monitorear la precisión de los modelos cada 3 meses y ajustar los hiperparámetros.
    """)

# Filtrar los datos según el rango de fechas seleccionado
filtered_df = df[(df['fecha'] >= start_filter) & (df['fecha'] <= end_filter)]

# Dashboard principal dividido en pestañas
tab1, tab2, tab3 = st.tabs(["📊 Resumen", "📈 Gráficos Detallados", "📋 Datos"])

with tab1:
    # Mostrar KPIs en tarjetas en la parte superior
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_precision = filtered_df['precision_modelo'].iloc[-1]
        delta_precision = current_precision - filtered_df['precision_modelo'].iloc[0]
        st.metric(
            label="Precisión Actual del Modelo",
            value=f"{current_precision:.1f}%",
            delta=f"{delta_precision:.1f}%"
        )
    
    with col2:
        current_alerts = filtered_df['num_alertas'].iloc[-1]
        avg_alerts = filtered_df['num_alertas'].mean()
        st.metric(
            label="Alertas Generadas (Último Mes)",
            value=f"{int(current_alerts)}",
            delta=f"{int(current_alerts - avg_alerts)} vs promedio"
        )
    
    with col3:
        current_days = filtered_df['dias_anticipacion'].iloc[-1]
        delta_days = current_days - filtered_df['dias_anticipacion'].iloc[0]
        st.metric(
            label="Días de Anticipación",
            value=f"{current_days:.1f} días",
            delta=f"{delta_days:.1f} días"
        )
    
    # Gráfico principal - Vista general
    st.subheader("Vista General de Indicadores Clave")
    
    # Crear gráfico múltiple para comparar todas las métricas
    fig = go.Figure()
    
    # Precisión del modelo (eje izquierdo)
    fig.add_trace(go.Scatter(
        x=filtered_df['fecha'],
        y=filtered_df['precision_modelo'],
        mode='lines',
        name='Precisión del Modelo (%)',
        line=dict(color='blue', width=3)
    ))
    
    # Días de anticipación (eje izquierdo)
    fig.add_trace(go.Scatter(
        x=filtered_df['fecha'],
        y=filtered_df['dias_anticipacion'],
        mode='lines',
        name='Días de Anticipación',
        line=dict(color='purple', width=3)
    ))
    
    # Número de alertas (eje derecho)
    fig.add_trace(go.Scatter(
        x=filtered_df['fecha'],
        y=filtered_df['num_alertas'],
        mode='lines',
        name='Número de Alertas',
        line=dict(color='orange', width=3),
        yaxis='y2'
    ))
    
    # Línea vertical para separar histórico y proyección
    projection_date = pd.to_datetime('2025-02-01')
    if projection_date >= start_filter and projection_date <= end_filter:
        fig.add_vline(
            x=projection_date, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Inicio Proyección",
            annotation_position="top right"
        )
    
    # Configuración de ejes y leyenda
    fig.update_layout(
        title='Evolución de las Métricas Clave',
        xaxis_title='Fecha',
        yaxis_title='Precisión (%) / Días',
        yaxis2=dict(
            title='Número de Alertas',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Mostrar gráficos detallados según las métricas seleccionadas
    if "Precisión del Modelo" in selected_metrics:
        st.subheader("Evolución de la Precisión del Modelo Predictivo")
        
        precision_fig = go.Figure()
        
        # Añadir línea histórica
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
        if not historical_df.empty:
            precision_fig.add_trace(go.Scatter(
                x=historical_df['fecha'],
                y=historical_df['precision_modelo'],
                mode='lines+markers',
                name='Datos Históricos',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ))
        
        # Añadir línea de proyección
        projection_df = filtered_df[filtered_df['tipo'] == 'Proyección']
        if not projection_df.empty:
            precision_fig.add_trace(go.Scatter(
                x=projection_df['fecha'],
                y=projection_df['precision_modelo'],
                mode='lines+markers',
                name='Proyección',
                line=dict(color='red', dash='dash', width=3),
                marker=dict(size=8)
            ))
        
        # Añadir línea de meta (85% de precisión)
        precision_fig.add_trace(go.Scatter(
            x=[filtered_df['fecha'].min(), filtered_df['fecha'].max()],
            y=[85, 85],
            mode='lines',
            name='Meta (85%)',
            line=dict(color='green', dash='dot', width=2)
        ))
        
        precision_fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Precisión del Modelo (%)',
            yaxis=dict(range=[60, 90]),
            hovermode='x unified'
        )
        
        st.plotly_chart(precision_fig, use_container_width=True)
        
        # Explicación del gráfico de precisión
        st.markdown("""
        **Análisis de la Precisión del Modelo:**
        - La precisión se calcula comparando las predicciones contra eventos reales
        - Se mide utilizando métricas como Accuracy y ROC-AUC
        - La meta es alcanzar un 85% de precisión para finales de año
        - La línea roja punteada indica el inicio de las proyecciones (febrero 2025)
        """)
    
    if "Número de Alertas" in selected_metrics:
        st.subheader("Número de Alertas Generadas por Mes")
        
        alertas_fig = go.Figure()
        
        # Añadir barras históricas
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
        if not historical_df.empty:
            alertas_fig.add_trace(go.Bar(
                x=historical_df['fecha'],
                y=historical_df['num_alertas'],
                name='Alertas (Histórico)',
                marker_color='royalblue'
            ))
        
        # Añadir barras de proyección
        projection_df = filtered_df[filtered_df['tipo'] == 'Proyección']
        if not projection_df.empty:
            alertas_fig.add_trace(go.Bar(
                x=projection_df['fecha'],
                y=projection_df['num_alertas'],
                name='Alertas (Proyección)',
                marker_color='indianred'
            ))
        
        alertas_fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Número de Alertas',
            hovermode='x unified'
        )
        
        st.plotly_chart(alertas_fig, use_container_width=True)
        
        # Estadísticas de alertas
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Promedio de alertas mensuales", f"{filtered_df['num_alertas'].mean():.1f}")
        with col2:
            st.metric("Tendencia anual", f"{(filtered_df['num_alertas'].iloc[-1] - filtered_df['num_alertas'].iloc[0]) / len(filtered_df) * 12:.1f} alertas/año")
    
    if "Días de Anticipación" in selected_metrics:
        st.subheader("Días de Anticipación en la Detección de Riesgos")
        
        anticipacion_fig = go.Figure()
        
        # Añadir línea histórica
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
        if not historical_df.empty:
            anticipacion_fig.add_trace(go.Scatter(
                x=historical_df['fecha'],
                y=historical_df['dias_anticipacion'],
                mode='lines+markers',
                name='Histórico',
                line=dict(color='purple', width=3),
                marker=dict(size=8)
            ))
        
        # Añadir línea de proyección
        projection_df = filtered_df[filtered_df['tipo'] == 'Proyección']
        if not projection_df.empty:
            anticipacion_fig.add_trace(go.Scatter(
                x=projection_df['fecha'],
                y=projection_df['dias_anticipacion'],
                mode='lines+markers',
                name='Proyección',
                line=dict(color='darkorange', dash='dash', width=3),
                marker=dict(size=8)
            ))
        
        # Añadir línea de meta (20 días de anticipación)
        anticipacion_fig.add_trace(go.Scatter(
            x=[filtered_df['fecha'].min(), filtered_df['fecha'].max()],
            y=[20, 20],
            mode='lines',
            name='Meta (20 días)',
            line=dict(color='green', dash='dot', width=2)
        ))
        
        anticipacion_fig.update_layout(
            xaxis_title='Fecha',
            yaxis_title='Días de Anticipación',
            hovermode='x unified'
        )
        
        st.plotly_chart(anticipacion_fig, use_container_width=True)
        
        # Explicación de la métrica
        st.info("""
        **Importancia de los días de anticipación:**
        Los días de anticipación representan cuánto tiempo antes de un evento de riesgo el modelo puede predecirlo con precisión.
        Una mayor anticipación permite implementar medidas preventivas con más tiempo, reduciendo el impacto potencial del riesgo.
        """)

with tab3:
    # Mostrar los datos en formato tabular
    st.subheader("Datos del Análisis Predictivo")
    
    # Opciones para descargar los datos
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos como CSV",
        data=csv,
        file_name='datos_analisis_predictivo.csv',
        mime='text/csv',
    )
    
    # Mostrar tabla con datos formateados
    display_df = filtered_df.copy()
    display_df['fecha'] = display_df['fecha'].dt.strftime('%Y-%m-%d')
    display_df['precision_modelo'] = display_df['precision_modelo'].round(2).astype(str) + '%'
    display_df['dias_anticipacion'] = display_df['dias_anticipacion'].round(1).astype(str) + ' días'
    
    st.dataframe(display_df, use_container_width=True)

# Sección de ayuda desplegable
with st.expander("ℹ️ Cómo usar este dashboard"):
    st.markdown("""
    **Instrucciones:**
    1. **Filtros**: Use el panel lateral para seleccionar el rango de fechas y las métricas que desea visualizar.
    2. **Pestañas**: Navegue entre las diferentes pestañas para acceder a distintas vistas:
       - **Resumen**: Visión general de las métricas clave y KPIs actuales.
       - **Gráficos Detallados**: Análisis individual de cada métrica seleccionada.
       - **Datos**: Tabla con los datos completos y opción para descargar.
    3. **Interactividad**: Puede interactuar con los gráficos:
       - Pasar el cursor para ver detalles
       - Hacer zoom en áreas específicas
       - Descargar la vista actual como imagen
    """)
