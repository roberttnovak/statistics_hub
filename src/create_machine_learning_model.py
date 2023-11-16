"""
This script performs data importation from a database, preprocesses time series data,
and executes a machine learning model to make predictions. The parameters for these operations
are loaded from a JSON configuration file.

Script Structure:
1. Load configurations from JSON files.
2. Set up logging.
3. Import data from the database.
4. Preprocess time series data.
5. Execute the machine learning model and store the model and preprocessed data.

Configuration Parameters (via JSON file):
- Data import parameters (e.g., database credentials, SQL query).
- Preprocessing parameters (e.g., resampling frequency, interpolation method).
- Machine learning model parameters (e.g., model type, features, target).
- Storage parameters (e.g., paths to save the model and preprocessed data).
"""

import datetime
import json
import os
from pathlib import Path
import pandas as pd
from OwnLog import OwnLogger
from cleaning import prepare_dataframe_from_db, process_time_series_data
from predictions import create_model_machine_learning_algorithm
from sql_utils import data_importer
from own_utils import load_json

# ToDo: Revisar los warnings

# Path to config folder
config_folder_name = "config"
config_path = os.path.join("..", config_folder_name)

# Config of models parameters
config_models_folder = "models_parameters"
model_name = "KNeighborsRegressor"
config_model_parameters = load_json(os.path.join(config_path,config_models_folder), model_name)

# Definition of configs variables for logs 
config_log_filename = "config_logs"
config_logs_parameters = load_json(config_path, config_log_filename)

# Logger params
log_rotation = config_logs_parameters["log_rotation"]
max_bytes = config_logs_parameters["max_bytes"]
backup_count = config_logs_parameters["backup_count"]
when = config_logs_parameters["when"]

# Create instance of OwnLogger
own_logger = OwnLogger(
    log_rotation = log_rotation,
    max_bytes = max_bytes,
    backup_count = backup_count,
    when = when
)

logger = own_logger.get_logger()

# Get DataFrame from database
# Obtener DataFrame desde la base de datos
data_importer_automatic_importation = config_model_parameters["data_importer_automatic_importation"]
data_importer_creds_path = Path(config_model_parameters["data_importer_creds_path"])
data_importer_path_instants_data_saved = Path(config_model_parameters["data_importer_path_instants_data_saved"])
data_importer_database  = config_model_parameters["data_importer_database"]
data_importer_query = config_model_parameters["data_importer_query"]
data_importer_save_importation = config_model_parameters["data_importer_save_importation"]
data_importer_file_name = config_model_parameters["data_importer_file_name"]

df = data_importer(
    automatic_importation = data_importer_automatic_importation,
    creds_path = data_importer_creds_path,
    path_instants_data_saved = data_importer_path_instants_data_saved,
    database  = data_importer_database,
    query = data_importer_query,
    save_importation = data_importer_save_importation,
    file_name = data_importer_file_name,
    logger = logger  
)

# Normalise time serie

# Prepare DataFrame from database
prepare_dataframe_from_db_cols_for_query = config_model_parameters["prepare_dataframe_from_db_cols_for_query"]

df = prepare_dataframe_from_db(
    df = df,
    cols_for_query = prepare_dataframe_from_db_cols_for_query,
    logger = logger
)

# Processing time serie data: uniform frequency and interpolating missing values
preprocess_time_series_data_resample_freq =  config_model_parameters["preprocess_time_series_data_resample_freq"]
preprocess_time_series_data_aggregation_func =  config_model_parameters["preprocess_time_series_data_aggregation_func"]
preprocess_time_series_data_method =  config_model_parameters["preprocess_time_series_data_method"]
preprocess_time_series_data_outlier_cols = config_model_parameters["preprocess_time_series_data_outlier_cols"]

df_resampled_interpolated = process_time_series_data(
    df = df ,
    resample_freq = preprocess_time_series_data_resample_freq,
    aggregation_func = preprocess_time_series_data_aggregation_func,
    method = preprocess_time_series_data_method,
    outlier_cols = preprocess_time_series_data_outlier_cols,
    logger = logger
)

df_preprocessed = pd.pivot_table(df_resampled_interpolated.reset_index()[["timestamp","id_device","id_variable","value"]],
               index=["timestamp","id_device"],
               columns=["id_variable"]).reset_index()

df_preprocessed.columns = ["timestamp", "id_device", "y", "temperatura", "humedad", "tvoc", "presion", "siaq", "diaq"]


# Model parameters 
ini_train = config_model_parameters["ini_train"]
fin_train = config_model_parameters["fin_train"]
fin_test = config_model_parameters["fin_test"]
name_time_column = config_model_parameters["name_time_column"]
name_id_sensor_column = config_model_parameters["name_id_sensor_column"]
id_device = config_model_parameters["id_device"]
names_objective_variable = config_model_parameters["names_objective_variable"]
predictor = config_model_parameters["predictor"]
X_name_features = config_model_parameters["X_name_features"] if config_model_parameters["X_name_features"] is not None else list(set(df_preprocessed.columns)-set(['y','timestamp','id_device']))
Y_name_features = config_model_parameters["Y_name_features"]
n_lags = config_model_parameters["n_lags"]
n_predictions = config_model_parameters["n_predictions"]
lag_columns = config_model_parameters["lag_columns"] if config_model_parameters["lag_columns"] is not None else list(set(df_preprocessed.columns)-set(['y','timestamp','id_device'])) + ["y"]
scale_in_preprocessing = config_model_parameters["scale_in_preprocessing"]
path_to_save_model = config_model_parameters["path_to_save_model"]
path_to_save_model = Path(config_model_parameters["path_to_save_model"])
save_preprocessing = config_model_parameters["save_preprocessing"]
folder_name_model = config_model_parameters["folder_name_model"]
folder_name_preprocessed_data = config_model_parameters["folder_name_preprocessed_data"]
now_str = f"execution-time-{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
folder_name_time_execution = config_model_parameters["folder_name_time_execution"] if config_model_parameters["folder_name_time_execution"] is not None else now_str
machine_learning_model_args = config_model_parameters["machine_learning_model_args"]

model_machine_learning = create_model_machine_learning_algorithm(
    tidy_data = df_preprocessed,
    ini_train = ini_train,
    fin_train = fin_train,
    fin_test = fin_test,
    id_device = id_device,
    model_sklearn_name = predictor,
    X_name_features = X_name_features,
    Y_name_features = Y_name_features,
    n_lags = n_lags,
    n_leads = n_predictions,
    lag_columns = lag_columns,
    lead_columns = Y_name_features,
    scale_in_preprocessing = scale_in_preprocessing,
    name_time_column = name_time_column,
    name_id_sensor = name_id_sensor_column,
    save_preprocessing = save_preprocessing,
    path_to_save_model= path_to_save_model,
    folder_name_model= folder_name_model,
    folder_name_time_execution = folder_name_time_execution,
    folder_name_preprocessed_data = folder_name_preprocessed_data,
    machine_learning_model_args = machine_learning_model_args
)