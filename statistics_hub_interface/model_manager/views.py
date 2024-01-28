import json
import logging
import os
from pathlib import Path
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed, HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
import sys
from django.shortcuts import render
from django.urls import reverse
from django.contrib.auth import authenticate, login
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import datetime
from django.views.decorators.clickjacking import xframe_options_exempt
 
sys.path.append(str(Path("../src")))
from visualisations import create_interactive_boxplot, create_interactive_plot, create_treeplot, plot_box_time_series, plot_weight_evolution
from PersistanceManager import PersistenceManager
from sql_utils import test_database_connection
from own_utils import filter_dataframe_by_column_values, load_json, modify_json_values
from ConfigManager import ConfigManager
from predictions import process_model_machine_learning, run_time_series_prediction_pipeline, evaluate_model
from eda import summary_statistics, summary_statistics_numerical


creds_path = '../global_creds/sql.json'

# ToDo: remove train_model
# ToDo: Use ConfigManager in all functions when necessary (for example upload_file instead manual modification)
# ToDo: Document better and apply good practice

@xframe_options_exempt
@login_required
def user_resources(request):
    return render(request, 'model_manager/user_resources.html')

@login_required
def user_resources_models(request):
    return render(request, 'model_manager/user_resources_models.html')

# ToDo: Deprecated, update
@login_required
def datasets(request):
    user = request.user
    pm = PersistenceManager(base_path=f"tenants/{user}", folder_datasets="data")
    datasets = pm.list_datasets()

    selected_dataset = None
    table_html = ""
    if request.method == 'POST':
        selected_dataset = request.POST.get('dataset')
        separator = request.POST.get('separator', ',')  # Se obtiene el separador o se usa ',' por defecto
        df = pm.load_dataset(selected_dataset.split(".")[0], csv_sep=separator)  # Asumiendo que load_dataset acepta un parámetro de separador
        table_html = df.head(1000).to_html(classes='table table-striped')

    context = {
        "datasets": datasets,
        "selected_dataset": selected_dataset,
        "table_html": table_html
    }
    return render(request, 'model_manager/datasets.html', context)


@login_required
def user_resources_models_saved(request):
    user = request.user
    pm = PersistenceManager(base_path=f"tenants/{user}/models")

    # Lista de modelos y sus rangos de entrenamiento.
    models_list = pm.get_available_models()
    
    # Diccionario para almacenar modelos, rangos y tiempos de ejecución.
    models_with_details = {}

    for model in models_list:
        ranges = pm.get_training_ranges_for_model(model)
        models_with_details[model] = {}

        for range_ in ranges:
            execution_times = pm.get_execution_times_for_model_and_range(model, range_)
            models_with_details[model][range_] = execution_times

    return render(request, 'model_manager/user_resources_models_saved.html', {
        'models_with_details': models_with_details
    })

@login_required
def model_evaluation_time_execution(request, model, training_range, execution_time):

    user = request.user
    path_config = f"tenants/{user}/config/models_parameters/common_parameters"
    config = load_json(path_config, "common_parameters")


    path_to_save_model = config["path_to_save_model"]
    folder_name_model = model
    folder_name_range_train = training_range
    folder_name_time_execution = execution_time
    pm = PersistenceManager(
        base_path=path_to_save_model,
        folder_name_model=folder_name_model,
        folder_name_range_train=folder_name_range_train, 
        folder_name_time_execution=folder_name_time_execution
    )

    # flags_exist = (path_to_predictions / flag_predictions_done).exists() and (path_to_predictions / flag_evaluations_done).exists()
    flag_predictions_done = config["flag_predictions_done"]
    flag_evaluations_done = config["flag_evaluations_done"]
    flag_predictions_done_exists = pm.flag_exists(flag_predictions_done)
    flag_evaluations_done_exists = pm.flag_exists(flag_evaluations_done)
    flags_exist = flag_predictions_done_exists and flag_evaluations_done_exists

    evaluation_files = pm.list_evaluations()

    active_view = request.POST.get('active_view', 'table-view')

    context = {
        'model': model,
        'training_range': training_range,
        'execution_time': execution_time,
        'flag_predictions_done_exists': flag_predictions_done_exists,
        'flag_evaluations_done_exists' : flag_evaluations_done_exists,
        'evaluation_files': evaluation_files,
        'active_view': active_view
    }


    if request.method == 'POST':
        # Parameters from user
        action = request.POST.get('action')
        freq = request.POST.get('freq', None)
        group_by_timestamp = request.POST.get('group_by_timestamp', 'true') == 'true'
        selected_file = request.POST.get('selected_file')
        selected_figure = request.POST.get('selected_figure')

        #ToDo: Generalizar mejor esto:
        #ToDo: Homogenizar lógica de los .txt en predictions y evaluations 
        save = True
        folder_name_preprocessed_data = config["folder_name_preprocessed_data"]
        folder_name_predictions = config["folder_name_predictions"]
        name_column_objective_variable_of_scaler = "y"
        overwrite_predictions = False

        if not flag_predictions_done_exists:

            process_model_machine_learning(
                path_to_save_model = path_to_save_model,
                folder_name_model = folder_name_model,
                folder_name_range_train = folder_name_range_train,
                folder_name_time_execution = folder_name_time_execution,
                folder_name_preprocessed_data = folder_name_preprocessed_data,
                save = save,
                folder_name_predictions = folder_name_predictions,
                name_column_objective_variable_of_scaler = name_column_objective_variable_of_scaler,
                flag_predictions_done = flag_predictions_done,
                overwrite_predictions = overwrite_predictions
            )

        folder_name_predictions = config["folder_name_predictions"]
        folder_name_evaluation= config["folder_name_evaluation_data"]
        flag_content = ""
        flag_subfolder = None
        freq = None 
        show_args_in_save = True

        if action == 'generate_evaluation':
            evaluate_model(
                path_to_save_model= path_to_save_model,
                folder_name_model=folder_name_model,
                folder_name_range_train=folder_name_range_train,
                folder_name_time_execution=folder_name_time_execution,
                folder_name_predictions = folder_name_predictions,
                save=save,
                flag_name=flag_evaluations_done,
                flag_content=flag_content,
                flag_subfolder=flag_subfolder,
                freq = freq,
                folder_name_evaluation= folder_name_evaluation,
                show_args_in_save = show_args_in_save,
                # save_name = "evaluations-extended",
                group_by_timestamp = group_by_timestamp
            )

        if selected_file:
            try:
                # Suponiendo que 'load_evaluation_data' carga los datos del archivo
                table = pm.load_evaluation_data(selected_file.split(".")[0], folder_name_evaluation_data=folder_name_evaluation)
                table_html = table.to_html(classes='table table-striped')
            except Exception as e:
                table_html = f"Error: {e}"
            
            context['table_html'] = table_html
            return render(request, 'model_manager/model_evaluation_time_execution.html', context)
        
        if selected_figure:
            predictions_train = pm.load_predictions("predictions-train")
            predictions_train["dataset_type"] = "train"
            predictions_test = pm.load_predictions("predictions-test")
            predictions_test["dataset_type"] = "test" 
            predictions_train_test = pd.concat([predictions_train,predictions_test]).reset_index(drop=True)
            df_train = pm.load_preprocessed_data("df_train")
            df_test = pm.load_preprocessed_data("df_test")
            df_train_test = pd.concat([df_train,df_test]).reset_index(drop=True)
            if selected_figure == 'figure1':
                fig = plot_box_time_series(predictions_train_test, df_train_test)
            else:
                fig = go.Figure()
                # fig.update_layout(
                #     xaxis={'visible': False},
                #     yaxis={'visible': False},
                #     annotations=[
                #         {
                #             "text": "No hay datos para mostrar",
                #             "xref": "paper",
                #             "yref": "paper",
                #             "showarrow": False,
                #             "font": {
                #                 "size": 20
                #             }
                #         }
                #     ]
                # )
            plot_html = py.plot(fig, output_type='div')
            context['plot_html'] = plot_html
            return render(request, 'model_manager/model_evaluation_time_execution.html', context)

        # return redirect('model_evaluation_time_execution', model=model, training_range=training_range, execution_time=execution_time)


    return render(request, 'model_manager/model_evaluation_time_execution.html', context)

@login_required
def model_evaluation_train_range(request, training_range):
    # Lógica para manejar la vista del rango de entrenamiento
    return render(request, 'model_manager/model_evaluation_train_range.html')

@login_required
def model_evaluation_model(request, model):
    # Lógica para manejar la vista del modelo
    return render(request, 'model_manager/model_evaluation_model.html')

@login_required
def model_evaluation_all_models(request):
    user = request.user
    path_config = f"tenants/{user}/config/models_parameters/common_parameters"
    config = load_json(path_config, "common_parameters")

    user_path = f"tenants/{user}/models"
    pm = PersistenceManager(base_path=user_path)

    models_list = pm.get_available_models(include_predictions_only=True)
    selected_figure = request.POST.get('selected_figure')

    # Define el valor inicial de active_view
    active_view = 'figure-view' if selected_figure else request.POST.get('active_view', 'table-view')

    context = {
        'models_list': models_list,
        'active_view': active_view,
    }

    if request.method == 'POST':
        action = request.POST.get('action')
        selected_models = request.POST.getlist('selected_models')

        # Define la función para calcular los pesos inversos
        def compute_inverse_weighted_averages(mae_contributions):
            inverse_weights = [1.0 / mae for mae in mae_contributions if mae != 0]
            total_inverse_weight = sum(inverse_weights)
            normalized_weights = [iw / total_inverse_weight for iw in inverse_weights]
            return normalized_weights

        # Recalcular weights_train para cada solicitud POST
        predictions_train_lst = [PersistenceManager(**args).load_predictions("predictions-train") for args in pm.get_models_info_as_dict(include_predictions_only=True)]
        predictions_test_lst = [PersistenceManager(**args).load_predictions("predictions-test") for args in pm.get_models_info_as_dict(include_predictions_only=True)]
        
        predictions_all_models = pd.concat(predictions_train_lst + predictions_test_lst)
        if selected_models:
            predictions_all_models = predictions_all_models.query("model.isin(@selected_models)")

        # Código para calcular weights_train
        weights_by_n_predictions = predictions_all_models\
            .groupby(['n_prediction', 'dataset_type'])\
            .agg({'mae': [('sum_mae', 'sum')]}).reset_index()
        weights_by_n_predictions.columns = ['n_prediction', 'dataset_type', 'sum_mae']

        weights_by_n_predictions_and_model = predictions_all_models\
            .groupby(['n_prediction', 'dataset_type', 'model'])\
            .agg({'mae': [('sum_mae_model', 'sum')]}).reset_index()
        weights_by_n_predictions_and_model.columns = ['n_prediction', 'dataset_type', 'model', 'sum_mae_model']

        weights = weights_by_n_predictions_and_model.merge(weights_by_n_predictions, on=['n_prediction', 'dataset_type'])
        weights['weight_mae'] = weights['sum_mae_model'] / weights['sum_mae']
        weights['weight_model'] = weights.groupby(["n_prediction", "dataset_type"])["weight_mae"].apply(compute_inverse_weighted_averages).explode().reset_index(drop=True)
        weights_train = weights.query("dataset_type == 'train'")

        # Actualiza el contexto con los datos de las tablas
        context['predictions_all_models_html'] = predictions_all_models.head(1000).to_html(classes='table table-striped', index=False)
        context['weights_html'] = weights.to_html(classes='table table-striped', index=False)

        if action == 'generate_figure' and selected_figure:
            # Código para manejar la generación de figuras
            if selected_figure == 'figure1':
                fig = plot_weight_evolution(weights_train, 'weight_mae')
                context['figure_description'] = "Descripción de la Figura 1."
            elif selected_figure == 'figure2':
                fig = plot_weight_evolution(weights_train, 'weight_model')
                context['figure_description'] = "Descripción de la Figura 2."
            else:
                fig = None

            if fig:
                plot_html = py.plot(fig, output_type='div')
                context['plot_html'] = plot_html

        context['selected_figure'] = selected_figure

    return render(request, 'model_manager/model_evaluation_all_models.html', context)
    

@login_required
def load_dataset(request):
    user = request.user
    pm = PersistenceManager(base_path=f"tenants/{user}", folder_datasets="data")
    datasets = pm.list_datasets()

    if request.method == 'POST':
        selected_dataset = request.POST.get('dataset')

        # Si no es vista previa, procesa para redireccionar
        separator = request.POST.get('separator', ',')
        if selected_dataset:
            # Redirige a la vista preprocess_dataset con los parámetros incluidos en la URL
            return redirect('preprocess_dataset', selected_dataset=selected_dataset, separator=separator)
        
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        try:
            selected_dataset = request.GET.get('dataset')
            raw_preview = pm.load_csv_as_raw_string('data', selected_dataset, num_rows=10)
            return JsonResponse({'raw_preview_html': '<pre>' + raw_preview + '</pre>'})
        except FileNotFoundError as e:
            return JsonResponse({'error': str(e)}, status=500)


    # Renderiza la plantilla si no es una solicitud POST o si no se seleccionó un dataset
    return render(request, 'model_manager/load_dataset.html', {'datasets': datasets})

# preprocess_dataset backend logic
@login_required
def preprocess_dataset(request, selected_dataset, separator):

    user = request.user

    # Funciones auxiliares definidas dentro de preprocess_dataset
    def load_data():
        pm = PersistenceManager(base_path=f"tenants/{user}", folder_datasets="data")
        try:
            df = pm.load_dataset(selected_dataset.split(".")[0], csv_sep=separator)
            return df, None
        except Exception as e:
            return None, str(e)

    def generate_html(df, id_table, n_first_rows_to_show=100):
        html = df.head(n_first_rows_to_show).to_html(classes='table table-striped')
        html_with_id = html.replace('<table', f'<table id="{id_table}"', 1)

        return html_with_id

    def get_min_max_dates_from_dataset(df, timestamp_column):
        if timestamp_column in df.columns and not df[timestamp_column].empty:
            # Convertir la columna a datetime, si no está ya en ese formato
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')

            # Calcula la fecha mínima y máxima, omitiendo valores NaT generados por 'coerce'
            min_date = df[timestamp_column].min()
            max_date = df[timestamp_column].max()

            # Formatear las fechas mínima y máxima, si no son NaT (not a time)
            if pd.notnull(min_date) and pd.notnull(max_date):
                min_date = min_date.strftime("%Y-%m-%d")
                max_date = max_date.strftime("%Y-%m-%d")
                return min_date, max_date
            else:
                return None, None
        else:
            return None, None
    
    df, error = load_data()
    df_filtered = df.copy()

    fig_eda = go.Figure()
    eda_plot_html = py.plot(fig_eda, output_type='div')

    fig_preprocessing = go.Figure()
    preprocessing_plot_html = py.plot(fig_preprocessing, output_type='div')

    eda_results = pd.DataFrame({}) #summary_statistics(df,["id_device"])

    context = {
        # 'active_view': active_view,
        'selected_dataset': selected_dataset,
        'separator': separator,
        'eda_results_html' : generate_html(eda_results, id_table = 'eda-results-table'),
        'eda_plot_html' : eda_plot_html,
        "preprocessing_plot_html": preprocessing_plot_html
    }



    if not error:
        columns = df.columns.tolist()
        timestamp_column = request.POST.get('timestamp_column')
        date_range_user = request.POST.get('date_range')
        if date_range_user:
            min_date_user, max_date_user = date_range_user.split(" to ")
            df_filtered = df_filtered[((df_filtered[timestamp_column] > min_date_user) & (df_filtered[timestamp_column] < max_date_user))]
        else:
            min_date_user, max_date_user = None, None
        min_date_dataset, max_date_dataset = get_min_max_dates_from_dataset(df, timestamp_column) if timestamp_column else (None, None)
        context.update({"df_html": generate_html(df_filtered, id_table = 'data-table-preview'), "columns": columns, "min_date_dataset":min_date_dataset, "max_date_dataset":max_date_dataset})
        # context = handle_eda(df,context)

    else:
        context.update({"error": error})

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':

        if request.POST.get('action') == 'update_dataset':
            timestamp_column = request.POST.get('timestamp_column')
            date_range_user = request.POST.get('date_range')
            filters_data_json = request.POST.get('filters_data')

            if date_range_user:
                min_date_user, max_date_user = date_range_user.split(" to ")
                df_filtered = df_filtered[((df_filtered[timestamp_column] > min_date_user) & (df_filtered[timestamp_column] < max_date_user))]

            if filters_data_json:
                try:
                    filters_data = json.loads(filters_data_json)
                    df_filtered = filter_dataframe_by_column_values(df_filtered, filters_data)
                    
                except json.JSONDecodeError:
                    # Manejar el caso en que los datos de los filtros no sean un JSON válido
                    print("Error al decodificar los datos de los filtros")


            context.update(
                {
                    "df_html": generate_html(df_filtered, id_table = 'data-table-preview')
                }
            )
            response_data = context
            return JsonResponse(response_data)
        
        if request.POST.get('action') == 'update_table_eda':
            timestamp_column = request.POST.get('timestamp_column')
            date_range_user = request.POST.get('date_range')
            filters_data_json = request.POST.get('filters_data')
            eda_columns = json.loads(request.POST.get('eda_columns'))
            eda_groupby_columns = json.loads(request.POST.get('eda_groupby_columns'))

            if date_range_user:
                min_date_user, max_date_user = date_range_user.split(" to ")
                df_filtered = df_filtered[((df_filtered[timestamp_column] > min_date_user) & (df_filtered[timestamp_column] < max_date_user))]

            if filters_data_json:
                try:
                    filters_data = json.loads(filters_data_json)
                    df_filtered = filter_dataframe_by_column_values(df_filtered, filters_data)
                    
                except json.JSONDecodeError:
                    # Manejar el caso en que los datos de los filtros no sean un JSON válido
                    print("Error al decodificar los datos de los filtros")
            eda_results = summary_statistics(df = df_filtered, variables = eda_columns, groupby = eda_groupby_columns)
            print(eda_results)
            fig_eda = create_interactive_boxplot(df_filtered, 'id_variable', 'id_device', 'value')
            eda_plot_html = py.plot(fig_eda, output_type='div')

            context.update(
                {
                    "eda_results_html": generate_html(eda_results, id_table = 'eda-results-table'),
                    "eda_plot_html": eda_plot_html,
                    "df_html": generate_html(df_filtered, id_table = 'data-table-preview')
                }
            )
            response_data = context
            return JsonResponse(response_data)

        if request.POST.get('action') == 'update_plot_eda':
            visualization_type = request.POST.get('visualization_type')
            path_cols = json.loads(request.POST.get('path_columns')) # ['id_device','id_sensor'] #request.POST.getlist('path_columns')  
            value_col = request.POST.get('value_column')
            summary_metric = request.POST.get('summary_metric')
            if visualization_type == 'treemap':
                fig_eda = create_treeplot(df=df, path_cols=path_cols, value_col=value_col, summary_metric=summary_metric)
                eda_plot_html = py.plot(fig_eda, output_type='div')
                context.update({
                    "eda_plot_html": eda_plot_html
                })
            response_data = context
            return JsonResponse(response_data)

        if request.GET.get('action') == 'fetch_min_max_dates_from_dataset':
            timestamp_column = request.GET.get('timestamp_column')  
            if timestamp_column:
                min_date_dataset, max_date_dataset = get_min_max_dates_from_dataset(df, timestamp_column)
                context.update({'min_date_dataset': min_date_dataset, 'max_date_dataset': max_date_dataset})
                return JsonResponse(context)
            else:
                return JsonResponse({'error': 'Columna de timestamp no especificada'}, status=400)
        
        # Verificar si la solicitud AJAX es para obtener categorías únicas
        if request.GET.get('action') == 'get_uniques_categories_from_filters':
            column_name = request.GET.get('column_name')
            if column_name and column_name in df.columns:
                unique_categories = df[column_name].dropna().unique().tolist()
                return JsonResponse({'categories': unique_categories})
            else:
                return JsonResponse({'error': 'Columna no especificada o no encontrada'}, status=400)

        

    return render(request, 'model_manager/preprocess_dataset.html', context)

@login_required
def get_models_list(request):
    user = request.user
    config_manager = ConfigManager(f"tenants/{user}/config")
    models_list = config_manager.list_configs(subfolder="models_parameters")
    return models_list

@login_required
def get_model_parameters(request, model_type):
    user = request.user
    config_manager = ConfigManager(f"tenants/{user}/config")
    model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters")
    legible_names_of_parameters = config_manager.load_config("legible_names_of_parameters", subfolder="models_parameters/metadata")
    descriptions_of_parameters = config_manager.load_config("description_of_parameters", subfolder="models_parameters/metadata")
    response_data = {
        "model_parameters_default": model_parameters_default,
        "legible_names_of_parameters": legible_names_of_parameters,
        "descriptions_of_parameters": descriptions_of_parameters
    }
    return response_data


@login_required
def model_train(request, model_name):
    if request.method == 'POST':
        config_path = f"tenants/{request.user.username}/config"
        config_log_filename = "config_logs"

        try:
            run_time_series_prediction_pipeline(config_path, model_name, config_log_filename)
            messages.success(request, "Model trained succesfully")
        except Exception as e:
            messages.error(request, f"Error during training: {e}")

        return redirect('model_selection') 

    else:
        return redirect('model_selection')

@login_required
def model_selection(request):
    selected_model = request.GET.get('selected_model')

    if request.method == 'POST':
        selected_model = request.POST.get('model_type')
        action = request.POST.get('action')
        if action == 'show_parameters':
            return redirect('model_parameters', model_name=selected_model)
        elif action == 'train_model':
            config_path = f"tenants/{request.user.username}/config"
            #ToDo: Tratar en algún momento los logs 
            config_log_filename = None
            try:
                run_time_series_prediction_pipeline(config_path, selected_model, config_log_filename)
            except Exception as e:
                messages.error(request, f"Error during training: {e}")

            return redirect('../user_resources_models_saved', model_name=selected_model)

    models_list = get_models_list(request)
    return render(request, 'model_manager/model_selection.html', {
        'models_list': models_list,
        'selected_model': selected_model  
    })


@login_required
def model_parameters(request, model_name):
    """
    View for handling the display and update of model parameters.

    Parameters:
    - request: HttpRequest object containing metadata and form data for POST requests.
    - model_name: str, the name of the model whose parameters are being edited.

    Returns:
    - HttpResponse: Rendered HTML page for GET requests or a redirect for POST requests.
    """
    user = request.user

    # Instantiate the ConfigManager with the path to the user's tenant configuration.
    config_manager = ConfigManager(f"tenants/{user}/config")

    if request.method == 'POST':
        if not model_name:
            model_name = request.POST.get('model_type')  
        # Capture the form data as a dictionary.
        updated_parameters = {param: request.POST.get(param) for param in request.POST if param != 'csrfmiddlewaretoken'}

        try:
            # Update the model's configuration using the captured form data.
            config_manager.update_config(model_name, updated_parameters, subfolder="models_parameters")

            # Redirect to avoid post data resubmission if the user refreshes the page.
            # return redirect('../../model_selection', model_name=model_name)
            return HttpResponseRedirect(f"{reverse('model_selection')}?selected_model={model_name}")

        except FileNotFoundError as e:
            # Log the error and handle it as appropriate.
            # (Not shown: You should include logging here for the error e)
            messages.error(request, "An error occurred while updating the parameters. Please try again.")

    # For a GET request, load the existing parameters to display in the form.
    try:
        # Load the model's current parameters from the configuration.
        model_parameters = config_manager.load_config(model_name, subfolder="models_parameters")

        regressor_params = model_parameters["regressor_params"]
        time_series_args = model_parameters["time_series_args"]

        config_manager_metadata_parameters = ConfigManager(f"../config")

        # Get metadata of sklearn regressor
        all_sklearn_regressors_with_all_info = config_manager_metadata_parameters.load_config("all_sklearn_regressors_with_all_info", subfolder="models_parameters/metadata")
        regressor_info = all_sklearn_regressors_with_all_info[model_name]['regressor_info']
        url_scrapped = all_sklearn_regressors_with_all_info[model_name]['url_scrapped']
        sklearn_parameters_info = all_sklearn_regressors_with_all_info[model_name]['parameters_info'] 

        # Get metadata of other parameters 
        legible_names_of_own_parameters = config_manager_metadata_parameters.load_config("legible_names_of_own_parameters", subfolder="models_parameters/metadata")
        descriptions_of_own_parameters = config_manager_metadata_parameters.load_config("descriptions_of_own_parameters", subfolder="models_parameters/metadata")
        metadata_of_own_parameters = [
        {
            "name": param,
            "legible_name": legible_names_of_own_parameters.get(param, param),  # Usa el nombre del parámetro como respaldo
            "description": descriptions_of_own_parameters.get(param, ""),
            "value": value
        }
        for param, value in model_parameters.items()
        ]
        context = {
                'model_name': model_name,
                'model_parameters': model_parameters,
                'regressor_info': regressor_info,
                'url_scrapped': url_scrapped,
                'sklearn_parameters_info': sklearn_parameters_info,
                'legible_names_of_own_parameters': legible_names_of_own_parameters,
                'descriptions_of_own_parameters' : descriptions_of_own_parameters,
                'metadata_of_own_parameters': metadata_of_own_parameters
            }

    except FileNotFoundError:
        # If the config file is not found, handle the error appropriately (e.g., log it, send an error message to the template, etc.)
        model_parameters = {}

    # Render the page with the current model parameters.
    return render(request, 'model_manager/model_parameters.html', context)

def data_source_selection(request):
    return render(request, 'model_manager/data_source_selection.html')

def connect_to_database(request):

    message = ''
    connection_successful = False

    if request.method == 'POST':

        ssh_host = request.POST['ssh_host']
        ssh_port = request.POST['ssh_port']
        ssh_user = request.POST['ssh_user']
        ssh_password = request.POST['ssh_password']
        db_server = request.POST['db_server']
        db_user = request.POST['db_user']
        db_password = request.POST['db_password']
        db = request.POST['db']

        changes = {
            'ssh_host': ssh_host,
            'ssh_port': ssh_port,
            'ssh_user': ssh_user,
            'ssh_password': ssh_password,
            'db_server': db_server,
            'db_user': db_user,
            'db_password': db_password
        }

        modify_json_values(creds_path, changes)

        with open(creds_path, 'r') as file:
            default_values = json.load(file)

        connection_successful = test_database_connection(
            ssh_host, int(ssh_port), ssh_user, ssh_password,
            db_server, db_user, db_password, db
        )
        message = 'Connection successful!' if connection_successful else 'Failed to connect.'
        
    else:
        # Read the default values for GET request
        with open(creds_path, 'r') as file:
            default_values = json.load(file)        
    
    return render(request, 
        'model_manager/connect_to_database.html', 
        {
            'default_values': default_values,
            'message': message,
            'connection_successful': connection_successful
        }
    )

def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirigir a la página de inicio después del login exitoso
            return redirect('user_resources')  # Asegúrate de reemplazar 'home' con el nombre de tu vista de inicio
        else:
            # Devolver al formulario de login con un mensaje de error
            return render(request, 'model_manager/login.html', {'form': { 'errors': True }})
    else:
        return render(request, 'model_manager/login.html')
    
@login_required  # Ensures that only authenticated users can access this view.
def upload_file(request):
    """
    Handles file uploads by authenticated users, displaying an upload form for GET requests
    and processing the file for POST requests. After a successful file upload, the user is
    redirected to avoid duplicate submissions if the page is refreshed. If a file is selected
    for training, 'data_source.json' is created or updated with the selected file's name,
    and the user is redirected to the '' view.

    Parameters:
    - request: HttpRequest object containing metadata, the uploaded file, or the selected file name.

    Returns:
    - HttpResponse: Redirect to the upload page to display the form and list of uploaded files,
                    redirect to the train model page after selecting a file, or a 405 response for methods not allowed.
    """
    if request.method == 'POST':
        if 'local_file' in request.FILES:
            # Handle file upload on POST request.
            user = request.user

            # Define the path to the 'data' directory within the user's tenant directory.
            tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', user.username, 'data')
            
            # Create the directory if it doesn't exist.
            os.makedirs(tenant_data_dir, exist_ok=True)

            # Use FileSystemStorage with the path to the tenant's 'data' directory.
            fs = FileSystemStorage(location=tenant_data_dir)

            # Get the uploaded file from the request.
            myfile = request.FILES['local_file']

            # Check for directory traversal in filename.
            if '..' in myfile.name or '/' in myfile.name:
                raise ValueError("Invalid filename.")

            try:
                # Save the file.
                filename = fs.save(myfile.name, myfile)
                
                # Get the URL of the saved file.
                uploaded_file_url = fs.url(filename)

                # Redirect to the same view to show the form and file list.
                return HttpResponseRedirect(reverse('upload_file'))
            except Exception as e:
                # Handle any unexpected exceptions during file save operation.
                # Log the exception and return an appropriate HTTP response.
                # (Logging not shown here, but should be implemented.)
                return HttpResponse(str(e), status=500)
        
        elif 'selected_file' in request.POST:
            # File selection logic for training
            selected_file = request.POST['selected_file']
            if '..' in selected_file or '/' in selected_file:
                # Additional security check
                raise ValueError("Invalid filename.")
            tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', request.user.username)
            data_source_path = os.path.join(tenant_data_dir, 'data_source.json')
            os.makedirs(tenant_data_dir, exist_ok=True)  # Ensure the directory exists
            with open(data_source_path, 'w') as file:
                # Write the selected file name to 'data_source.json'
                json.dump({'selected_file': selected_file}, file)
            return redirect('model_selection')  # Redirect to the training view
        else:
            # Invalid POST request
            return HttpResponse("Invalid POST request.", status=400)

    elif request.method == 'GET':
        # Show the upload form on GET request.
        user = request.user

        # Define the path to the 'data' directory within the user's tenant directory.
        tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', user.username, 'data')

        # Use FileSystemStorage to list files in the directory.
        fs = FileSystemStorage(location=tenant_data_dir)

        # List files already uploaded.
        user_files = [{'name': file, 'url': fs.url(file)} for file in fs.listdir('')[1]]

        # Render the upload form and list of files.
        return render(request, 'model_manager/upload_file.html', {
            'user_files': user_files
        })

    else:
        # Return a response indicating that the method is not allowed.
        # Only POST and GET are supported.
        return HttpResponseNotAllowed(['POST', 'GET'])
    

