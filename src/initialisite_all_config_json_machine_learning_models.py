"""
This script initializes and configuration for regression models of sklearn.

It performs the following operations:
1. Imports necessary functions and classes from other modules.
2. Retrieves a list of regressors along with their parameters.
3. Constructs a template dictionary of initial parameters for configuration.
4. Creates an instance of ConfigManager to manage configurations.
5. Saves the list of regressors with their parameters as a configuration.
6. Saves the initial configuration for each regressor.
"""

from predictions import get_all_regressors_with_its_parameters
from ConfigManager import ConfigManager


all_regressors_with_its_parameters = get_all_regressors_with_its_parameters()

all_regressors = list(all_regressors_with_its_parameters.keys())

# Template of json to save in all models 
initial_dict_parameters = {
    'data_importer_automatic_importation': False, 
    'data_importer_creds_path': '../creds/sql.json', 
    'data_importer_path_instants_data_saved': '../data/instants_data_saved', 
    'data_importer_database': None, 
    'data_importer_query': None, 
    'data_importer_save_importation': True, 
    'data_importer_file_name': None, 
    'prepare_dataframe_from_db_cols_for_query': ['00-eco2', '00-temp', '01-hum', '01-tvoc', '02-pres', '03-siaq', '04-diaq'], 
    'preprocess_time_series_data_resample_freq': '60S', 
    'preprocess_time_series_data_aggregation_func': 'mean', 
    'preprocess_time_series_data_method': 'linear', 
    'preprocess_time_series_data_outlier_cols': None, 
    'ini_train': '2023-04-18 00:00:00+00:00', 
    'fin_train': '2023-04-25 00:00:00+00:00',
    'fin_test': '2023-04-26 00:00:00+00:00', 
    'name_time_column': 'timestamp', 
    'name_id_sensor_column': 'id_device', 
    'id_device': 'DBEM003', 
    'names_objective_variable': 'y', 
    'predictor': 'KNeighborsRegressor', 
    'X_name_features': None,
    'Y_name_features': 'y', 
    'n_lags': 10, 
    'n_predictions': 20, 
    'lag_columns': None, 
    'lead_columns': 'y', 
    'num_obs_to_predict': None, 
    'scale_in_preprocessing': True, 
    'path_to_save_model': '../models', 
    'save_preprocessing': True, 
    'folder_name_model': None, 
    'folder_name_preprocessed_data': 'preprocessed-data-to-use-in-model', 
    'folder_name_time_execution': None, 
    'machine_learning_model_args': {}}

# Initialisation of a instance of ConfigManager
config_manager = ConfigManager("../config")


# Save parameters allowed in each regressor
config_manager.save_config(
    "all_regresors_with_its_parameters", 
    all_regressors_with_its_parameters, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save all config
[config_manager.save_config(regressor, initial_dict_parameters, subfolder = "models_parameters") 
 for regressor in all_regressors]

# Update predictor parameter according to the regressor
[config_manager.update_config(regressor, {"predictor":regressor}, subfolder = "models_parameters") 
 for regressor in all_regressors]