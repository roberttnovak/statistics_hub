import plotly.express as px
import plotly.graph_objects as go

def plot_box_time_series(predictions_train_test, df_train_test):
    """
    This function creates a visualization containing a boxplot and a time series plot.
    
    The boxplot represents the distribution of the Mean Absolute Error (MAE) 
    for training and testing datasets, while the time series plot represents
    the real values over time.
    
    Parameters:
    - predictions_train_test (pd.DataFrame): A DataFrame containing predictions 
        with a 'timestamp_real', 'mae', and 'dataset_type' columns.
    - df_train_test (pd.DataFrame): A DataFrame containing real values with a 
        'timestamp' and 'y' columns.
    
    Returns:
    - fig (plotly.graph_objects.Figure): A Figure object displaying the boxplot and 
        the time series plot.
        
    Examples:
    >>> plot_box_time_series(predictions_train_test, df_train_test, tidy_data).show()
    """

    # Update dataset_type values for better labeling
    updated_predictions = predictions_train_test.replace({'dataset_type': {'train': 'boxplots train', 'test': 'boxplots test'}})
    
    # Creating the boxplot
    box_fig = px.box(
        updated_predictions, 
        x="timestamp_real", 
        y='mae', 
        title='Boxplot of MAE', 
        color="dataset_type"
    )

    # Creating the time series
    time_series_fig = px.line(
        df_train_test.assign(legend=lambda x: "time serie"), 
        x='timestamp', 
        y='y', 
        title='Time Series of Real Values',
        color='legend'
    )
    time_series_fig.update_traces(line_color='black')  # Updating line color to black for consistency

    # Creating an empty figure
    fig = go.Figure()

    # Adding the boxplot to the figure
    for trace in box_fig.data:
        trace.yaxis = 'y2'  # Assigning the boxplot to Y2 axis (right)
        fig.add_trace(trace)

    # Adding the time series to the figure
    for trace in time_series_fig.data:
        fig.add_trace(trace)

    # Updating the layout of the figure to include separate Y axes and a title
    fig.update_layout(
        yaxis=dict(
            title='Value',
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        yaxis2=dict(
            title='MAE',
            titlefont=dict(
                color="#ff7f0e"
            ),
            tickfont=dict(
                color="#ff7f0e"
            ),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        title_text="Boxplot and Time Series Visualization"
    )

    return fig

def plot_weight_evolution(df, weight_column):
    """
    Generate a stacked bar plot to visualize the evolution of model weights over observations.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    weight_column (str): The name of the column in the DataFrame that contains the weight values to be plotted.

    Returns:
    fig (plotly.graph_objs._figure.Figure): Plotly Figure object that can be displayed in a Jupyter notebook.
    """

    # Ordenar los datos
    df_sorted = df.sort_values(by=[weight_column], ascending=True)

    # Creando el gráfico
    fig = px.bar(df_sorted, 
                 x="n_prediction", 
                 y=weight_column, 
                 color="model", 
                 title=f"Evolution of the {weight_column} along n predictions",
                 labels={"n_prediction": "n prediction", weight_column: "Weights"},
                 barmode='stack')

    return fig