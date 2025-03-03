import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import datetime

# Crear datos de muestra
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

# Crear aplicación Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Análisis Predictivo de Riesgos Emergentes", 
            style={'textAlign': 'center', 'marginBottom': 30, 'marginTop': 20}),
    
    # Resumen OKR
    html.Div([
        html.H2("Objetivo del OKR:", style={'marginBottom': 10}),
        html.P("Implementar un sistema de análisis predictivo para riesgos emergentes en los próximos 12 meses."),
        
        html.H3("Key Results (KR):", style={'marginBottom': 10, 'marginTop': 20}),
        html.Ul([
            html.Li("KR1: Desarrollar 5 modelos predictivos de riesgo basados en datos de clientes y mercados en 6 meses."),
            html.Li("KR2: Aumentar la precisión de predicción de riesgos en un 20% en 12 meses."),
            html.Li("KR3: Integrar el 100% de los datos históricos relevantes en el sistema de análisis predictivo.")
        ])
    ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),
    
    # Filtros y controles
    html.Div([
        html.H3("Filtros y Controles:", style={'marginBottom': 15}),
        html.Div([
            html.Label("Rango de Fechas:"),
            dcc.DatePickerRange(
                id='date-picker-range',
                min_date_allowed=start_date,
                max_date_allowed=end_date,
                start_date=start_date,
                end_date=end_date,
                style={'marginBottom': 20}
            ),
        ]),
        html.Div([
            html.Label("Seleccionar Métricas:"),
            dcc.Checklist(
                id='metrics-checklist',
                options=[
                    {'label': 'Precisión del Modelo', 'value': 'precision'},
                    {'label': 'Número de Alertas', 'value': 'alertas'},
                    {'label': 'Días de Anticipación', 'value': 'anticipacion'}
                ],
                value=['precision', 'alertas', 'anticipacion'],
                inline=True
            ),
        ]),
    ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),
    
    # Gráficos
    html.Div([
        html.Div([
            html.H3("Evolución de la Precisión del Modelo Predictivo", style={'textAlign': 'center'}),
            dcc.Graph(id='precision-graph')
        ], style={'marginBottom': 30}),
        
        html.Div([
            html.H3("Número de Alertas Generadas por Mes", style={'textAlign': 'center'}),
            dcc.Graph(id='alertas-graph')
        ], style={'marginBottom': 30}),
        
        html.Div([
            html.H3("Días de Anticipación en la Detección de Riesgos", style={'textAlign': 'center'}),
            dcc.Graph(id='anticipacion-graph')
        ]),
    ]),
    
    # Resumen de acciones necesarias
    html.Div([
        html.H3("Acciones Necesarias:", style={'marginBottom': 15}),
        html.Ul([
            html.Li("Entrenar y mejorar los modelos predictivos con datos más recientes y relevantes."),
            html.Li("Optimizar el pipeline de datos para mejorar la integración de fuentes externas (mercado, clima, geolocalización)."),
            html.Li("Monitorear la precisión de los modelos cada 3 meses y ajustar los hiperparámetros según los resultados obtenidos.")
        ])
    ], style={'marginTop': 30, 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '10px'}),
])

# Callbacks para actualizar los gráficos
@app.callback(
    [Output('precision-graph', 'figure'),
     Output('alertas-graph', 'figure'),
     Output('anticipacion-graph', 'figure')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('metrics-checklist', 'value')]
)
def update_graphs(start_date, end_date, selected_metrics):
    # Filtrar los datos por fecha
    filtered_df = df[(df['fecha'] >= start_date) & (df['fecha'] <= end_date)]
    
    # Gráfico de precisión del modelo
    precision_fig = go.Figure()
    
    if 'precision' in selected_metrics:
        # Añadir línea histórica
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
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
        hovermode='x unified',
        legend=dict(y=0.99, x=0.01, orientation='h'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    # Gráfico de número de alertas
    alertas_fig = go.Figure()
    
    if 'alertas' in selected_metrics:
        # Añadir barras históricas
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
        alertas_fig.add_trace(go.Bar(
            x=historical_df['fecha'],
            y=historical_df['num_alertas'],
            name='Alertas (Histórico)',
            marker_color='royalblue'
        ))
        
        # Añadir barras de proyección
        projection_df = filtered_df[filtered_df['tipo'] == 'Proyección']
        alertas_fig.add_trace(go.Bar(
            x=projection_df['fecha'],
            y=projection_df['num_alertas'],
            name='Alertas (Proyección)',
            marker_color='indianred'
        ))
    
    alertas_fig.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Número de Alertas',
        hovermode='x unified',
        legend=dict(y=0.99, x=0.01, orientation='h'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    # Gráfico de días de anticipación
    anticipacion_fig = go.Figure()
    
    if 'anticipacion' in selected_metrics:
        # Añadir línea histórica
        historical_df = filtered_df[filtered_df['tipo'] == 'Histórico']
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
        hovermode='x unified',
        legend=dict(y=0.99, x=0.01, orientation='h'),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    
    return precision_fig, alertas_fig, anticipacion_fig

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
