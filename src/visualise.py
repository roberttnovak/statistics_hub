import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import os
from PersistanceManager import PersistenceManager

#ToDo: ACtualizar esto en algún momento con testeos.ipynb

path_to_save_model = os.path.join("..","models")

folder_name_model = "KNeighborsRegressor"
folder_name_range_train = "initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0"
folder_name_time_execution = "execution-time-2023_10_26_13_31_40" 
folder_name_predictions = "predictions"

pm = PersistenceManager(
    base_path = path_to_save_model,
    folder_name_model=folder_name_model,
    folder_name_range_train = folder_name_range_train,
    folder_name_time_execution=folder_name_time_execution
)

predictions_train = pm.load_predictions("predictions-train")
predictions_test = pm.load_predictions("predictions-test")
app = dash.Dash(__name__)

def create_figure(df):
    """
    Crea una figura interactiva usando Plotly.

    Parameters:
    - df (pd.DataFrame): El DataFrame que contiene tus datos.

    Returns:
    - fig (go.Figure): La figura interactiva de Plotly.
    """
    # Crear el trazo para el valor real
    trace_real = go.Scatter(
        x=df['timestamp_real'],
        y=df['value_real'],
        mode='lines',
        name='Valor Real'
    )

    # Crear un trazo vacío para las predicciones, que se llenará al hacer clic
    trace_pred = go.Scatter(
        x=[None],
        y=[None],
        mode='lines+markers',
        name='Predicción'
    )

    fig = go.Figure(data=[trace_real, trace_pred])

    fig.update_layout(
        title='Visualización de Predicciones',
        xaxis_title='Timestamp',
        yaxis_title='Valor',
        template='plotly_dark',
        updatemenus=[dict(
            type='buttons',
            buttons=list([
                dict(label='Seleccione un punto',
                     method='relayout',
                     args=[{'title': 'Seleccione un punto en la línea de Valor Real para ver las predicciones'}])
            ]),
            direction='left',
            pad={'r': 10, 't': 10},
            showactive=True,
            x=0.17,
            xanchor='right',
            y=1.15,
            yanchor='top'
        )]
    )

    fig.update_traces(marker=dict(size=10),
                      selector=dict(mode='markers'))

    return fig

@app.callback(
    dash.dependencies.Output('prediction-plot', 'figure'),
    [dash.dependencies.Input('real-value-plot', 'clickData')]
)
def update_trace(df, clickData):
    if clickData is None:
        return go.Figure()
    else:
        idx = clickData['points'][0]['pointIndex']
        timestamp_selected = df['timestamp_real'].iloc[idx]
        df_pred = df[df['timestamp_real'] == timestamp_selected]

        return go.Figure(
            data=[
                go.Scatter(
                    x=df_pred['timestamp_prediction'],
                    y=df_pred['prediction'],
                    mode='lines+markers',
                    name='Predicción'
                )
            ],
            layout=go.Layout(
                title='Predicción',
                xaxis=dict(title='Timestamp'),
                yaxis=dict(title='Valor')
            )
        )

app.layout = html.Div([
    dcc.Graph(
        id='real-value-plot',
        figure=create_figure(predictions_test)
    ),
    dcc.Graph(
        id='prediction-plot'
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
