from django_plotly_dash import DjangoDash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

# Datos de ejemplo para los gráficos
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [40, 10, 20, 20, 40, 50],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Dashboard 1: Gráfico de barras
app1 = DjangoDash('Dashboard1')
app1.layout = html.Div(
    [
        dcc.Graph(
            id='graph-1',
            figure=px.bar(df, x="Fruit", y="Amount", color="City", barmode="group"),
            style={'height': '300vh', 'width': '100%'}  # Estilos para el componente Graph
        )
    ],
    style={'height': '300vh', 'width': '100%', 'padding-bottom': '70%'}  # Estilos para el componente Div
)

# Dashboard 2: Gráfico de dispersión
app2 = DjangoDash('Dashboard2')
app2.layout = html.Div(
    [
        html.H1(
            "Dashboard 2: Gráfico de Dispersión",
            style={'textAlign': 'center'}
        ),
        dcc.Graph(
            id='graph-2',
            figure=px.scatter(df, x="Fruit", y="Amount", color="City"),
            style={'height': '90vh', 'width': '100%'}  # Estilos para el componente Graph
        )
    ],
    style={'height': '90vh', 'width': '100%'}  # Estilos para el componente Div
)

# Más dashboards según sea necesario
