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


def create_interactive_plot(time_var, y_vars, df):
    """
    Create an interactive Plotly plot with buttons to switch between different Y variables.

    Parameters:
    - time_var (str): The name of the column to be used as the X axis (time variable).
    - y_vars (list of str): A list of column names to be used as Y variables.
    - df (pandas.DataFrame): The dataframe containing the data.

    Returns:
    - A Plotly figure with interactive buttons to switch between Y variables.

    Example:
    >>> create_interactive_plot('Time', ['Temperature', 'Humidity'], df)
    """

    # Initial trace
    trace = [go.Scatter(x=df[time_var], y=df[y_vars[0]], mode='lines', name=y_vars[0])]

    # Create figure with initial trace
    fig = go.FigureWidget(data=trace)

    # Create buttons for each Y variable
    buttons = []

    for var in y_vars:
        buttons.append(dict(method='restyle',
                            label=var,
                            visible=True,
                            args=[{'y': [df[var]],
                                   'name': var},
                                  [0]]))

    # Update layout with buttons
    fig.update_layout(
        updatemenus=[dict(active=0,
                          buttons=buttons,
                          direction="down",
                          pad={"r": 10, "t": 10},
                          showactive=True,
                          x=0.1,
                          xanchor="left",
                          y=1.1,
                          yanchor="top")]
    )

    return fig

import plotly.graph_objs as go

def create_interactive_boxplot(df, category_col, subcategory_col, value_col):
    """
    Create an interactive boxplot using Plotly.

    This function generates a boxplot for each unique combination of a category
    and subcategory in the provided dataframe. It includes interactive
    buttons to toggle between different categories.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data to plot.
    category_col (str): The name of the column in df representing the main category.
    subcategory_col (str): The name of the column in df representing the subcategory.
    value_col (str): The name of the column in df representing the values to plot.

    Returns:
    go.Figure: The Plotly figure object with the interactive boxplot.

    Example:
    >>> import pandas as pd
    >>> # Crear un DataFrame con más filas para mostrar distintas combinaciones de variables e id_beacons
    >>> df = pd.DataFrame({
    >>>     'variable': ['temp', 'temp', 'hum', 'hum', 'lux', 'lux'],
    >>>     'id_beacon': [1, 2, 1, 2, 1, 2],
    >>>     'value': [23, 24, 45, 50, 300, 350]
    >>> })
    >>> # Crear y mostrar el gráfico boxplot interactivo
    >>> fig = create_interactive_boxplot(df, 'variable', 'id_beacon', 'value')
    >>> fig.show()
    """

    # Create an empty figure
    fig = go.Figure()

    # Get the unique values of the category and subcategory columns
    categories = df[category_col].unique()
    subcategories = df[subcategory_col].unique()

    # Add a boxplot for each category and subcategory
    for category in categories:
        for subcategory in subcategories:
            # Filter the dataframe for the current category and subcategory
            filtered_df = df[(df[category_col] == category) & (df[subcategory_col] == subcategory)]
            fig.add_trace(go.Box(y=filtered_df[value_col],
                                 name=f"{category} - {subcategory}",
                                 visible=False))

    # Make the first set of boxplots visible by default
    for i in range(len(subcategories)):
        fig.data[i].visible = True

    # Create legend buttons for each category
    legend_buttons = [dict(label=category,
                           method='update',
                           args=[{'visible': [c == category for c in categories for _ in subcategories]}])
                      for category in categories]

    # Update the layout with custom legend
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                buttons=legend_buttons,
                x=1.1,
                xanchor='left',
                y=1,
                yanchor='top'
            )
        ],
        legend=dict(
            x=0.0,
            y=1.0,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
        )
    )

    return fig


def create_treeplot(df, path_cols, value_col=None, summary_metric='mean'):
    """
    Create an interactive treeplot using Plotly, displaying a hierarchy of nodes based on specified path columns.
    The function calculates various statistics for the given value column and displays these statistics on the 
    most granular level of the treeplot. It allows for an interactive exploration of data at different hierarchical levels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data. This DataFrame should have the columns specified in 'path_cols' 
                       and, if provided, the 'value_col'.
    path_cols (list of str): A list of column names that define the hierarchy for the treeplot. 
                             These columns will determine the structure and levels of the treeplot.
    value_col (str, optional): The name of the column in 'df' for which various statistics (like mean, median, quartiles, etc.) 
                               will be calculated. If None, the function will count the occurrences at each hierarchical level.
                               Default is None.
    summary_metric (str): A specific metric from the calculated statistics to determine the size of each node in the treeplot.
                          Common choices include 'mean', 'median', 'count', etc. This is only relevant if 'value_col' is provided.
                          Default is 'mean'.

    Returns:
    plotly.graph_objs._figure.Figure: A Plotly figure object representing the interactive treeplot.

    Examples:
    # Example DataFrame
    data = {'id_device': ['DBEM003', 'DBEM003'], 'id_sensor': ['sWEA', 'sWEA'],
            'id_variable': ['00-temp', '00-temp'], 'value': [18.57, 18.56]}
    df = pd.DataFrame(data)

    # Create a treeplot with counts (no value_col provided)
    fig = create_treeplot(df, ['id_device', 'id_sensor', 'id_variable'])
    fig.show()

    # Create a treeplot with value statistics (value_col provided)
    fig = create_treeplot(df, ['id_device', 'id_sensor', 'id_variable'], value_col='value')
    fig.show()

    Notes:
    - The function is designed to provide an overview of data distribution and statistics in a hierarchical format.
    - The interactive nature of the treemap allows users to explore different levels of the hierarchy easily.
    - The hover text on the most granular level nodes provides detailed statistical information, 
      offering insights into the underlying data distribution.
    """
    # Ensure path columns exist in the DataFrame
    for col in path_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    # Determine the aggregation function based on the presence of a value column
    if value_col:
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")
        # Aggregate using the specified value column
        df_agg = df.groupby(path_cols)[value_col].agg(
            count=('count'),
            Q1=(lambda x: x.quantile(0.25)),
            Q3=(lambda x: x.quantile(0.75)),
            percentile_90=(lambda x: x.quantile(0.9)),
            percentile_99=(lambda x: x.quantile(0.99)),
            median=('median'),
            mean=('mean'),
            std=('std'),
            min=('min'),
            max=('max')
        ).reset_index()
    else:
        # Count occurrences if no value column is provided
        df_agg = df.groupby(path_cols).size().reset_index(name='count')

    # Creating the treeplot with all metrics in hover_data
    fig = px.treemap(df_agg, path=[px.Constant('All')] + path_cols, values=summary_metric, 
                     hover_data=df_agg.columns)
    return fig