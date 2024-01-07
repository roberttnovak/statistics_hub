from django_plotly_dash import DjangoDash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd

# Datos de ejemplo para los gráficos
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Dashboard 1: Gráfico de barras
app1 = DjangoDash('Dashboard1')
app1.layout = html.Div([
    html.H1("Dashboard 1: Gráfico de Barras"),
    dcc.Graph(
        id='graph-1',
        figure=px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    )
])

# Dashboard 2: Gráfico de dispersión
app2 = DjangoDash('Dashboard2')
app2.layout = html.Div([
    html.H1("Dashboard 2: Gráfico de Dispersión"),
    dcc.Graph(
        id='graph-2',
        figure=px.scatter(df, x="Fruit", y="Amount", color="City")
    )
])

# Más dashboards según sea necesario
