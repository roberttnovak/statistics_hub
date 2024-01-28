"""
This script initializes and configuration for regression models of sklearn.

It performs the following operations:
1. Imports necessary functions and classes from other modules.
2. Retrieves a list of regressors of sklearn along with their parameters and other information
3. Constructs a template dictionary of initial parameters for configuration.
4. Creates an instance of ConfigManager to manage configurations.
5. Saves the list of regressors with their parameters as a configuration.
6. Saves the initial configuration for each regressor.
"""
import sklearn 
import datetime
from own_utils import evaluate_or_return_default
from sklearn_utils import get_all_regressors, get_all_regressors_with_its_parameters, get_valid_regressors_info
from ConfigManager import ConfigManager



# Get all regressors names of scikit learn 
all_regressors = get_all_regressors()

# Get all parameters of each regressor of scikit learn
all_regressors_with_its_parameters = get_all_regressors_with_its_parameters()

# Scrap all regressor and parameters from oficial documentation of sklearn 
all_regressors_with_all_info = get_valid_regressors_info()

# Save metadata of version of sklearn and date when documentation is scrapped
documentation_scrapping_date = datetime.datetime.now().strftime("%Y-%m-%d")
sklearn_version = sklearn.__version__
sklearn_details = {
    "sklearn_version": sklearn_version,
    "documentation_scrapping_date": documentation_scrapping_date
}

# Template of json to save in all models 
initial_dict_parameters = {
    'data_importer_args' :
        {
            'data_importer_automatic_importation': False, 
            'data_importer_database': None, 
            'data_importer_query': None, 
            'data_importer_save_importation': True, 
            'data_importer_file_name': None
        }, 
    'preprocess_time_series_data_args':
        {
            'preprocess_time_series_data_resample_freq': '60S', 
            'preprocess_time_series_data_aggregation_func': 'mean', 
            'preprocess_time_series_data_method': 'linear', 
            'preprocess_time_series_data_outlier_cols': None
        },
    'split_train_test_args': {
        'ini_train': '2023-04-18 00:00:00+00:00',
        'fin_train': '2023-04-25 00:00:00+00:00',
        'fin_test': '2023-04-26 00:00:00+00:00'
    },
    'time_serie_args': {
        'name_time_column': 'timestamp',
        'name_id_sensor_column': 'id_device',
        'id_device': 'DBEM003', 
        'names_objective_variable': 'y', 
        'prepare_dataframe_from_db_cols_for_query': ['00-eco2', '00-temp', '01-hum', '01-tvoc', '02-pres', '03-siaq', '04-diaq'], 
        'X_name_features': None,
        'Y_name_features': 'y',
        'n_lags': 10,
        'n_predictions': 20,
        'lag_columns': None,
        'lead_columns': 'y',
        'num_obs_to_predict': None
    }, 
    'predictor': 'KNeighborsRegressor',
    'save_args':
    {
        'scale_in_preprocessing': True, 
        'save_preprocessing': True, 
        'folder_name_model': None, 
        'folder_name_time_execution': None 
    }
}

initial_dict_parameters_legible_names = {
    "data_importer_automatic_importation": "Automatic Data Importation",
    "data_importer_creds_path": "Credentials Path",
    "data_importer_path_instants_data_saved": "Instants Data Saved Path",
    "data_importer_database": "Database Name",
    "data_importer_query": "SQL Query",
    "data_importer_save_importation": "Save Imported Data",
    "data_importer_file_name": "Imported Data File Name",
    "prepare_dataframe_from_db_cols_for_query": "Columns for Query",
    "preprocess_time_series_data_resample_freq": "Resampling Frequency",
    "preprocess_time_series_data_aggregation_func": "Aggregation Function",
    "preprocess_time_series_data_method": "Interpolation Method",
    "preprocess_time_series_data_outlier_cols": "Outlier Columns",
    "ini_train": "Training Start Time",
    "fin_train": "Training End Time",
    "fin_test": "Testing End Time",
    "name_time_column": "Time Column Name",
    "name_id_sensor_column": "Sensor ID Column Name",
    "id_device": "Device ID",
    "names_objective_variable": "Objective Variable Name",
    "predictor": "Predictor Model",
    "X_name_features": "Feature Names",
    "Y_name_features": "Target Feature Name",
    "n_lags": "Number of Lags",
    "n_predictions": "Number of Predictions",
    "lag_columns": "Lag Columns",
    "lead_columns": "Lead Column",
    "num_obs_to_predict": "Number of Observations to Predict",
    "scale_in_preprocessing": "Scale in Preprocessing",
    "path_to_save_model": "Model Save Path",
    "save_preprocessing": "Save Preprocessing",
    "folder_name_model": "Model Folder Name",
    "folder_name_preprocessed_data": "Preprocessed Data Folder Name",
    "folder_name_time_execution": "Execution Time Folder Name",
    "machine_learning_model_args": "Machine Learning Model Arguments"
}

initial_dict_parameters_descriptions = {
    "data_importer_automatic_importation": "Boolean value to determine if the data importation should be executed automatically or not.",
    "data_importer_creds_path": "Path to the credentials file for database access.",
    "data_importer_path_instants_data_saved": "Path where instants data from the database are saved.",
    "data_importer_database": "Name of the database from which to import data.",
    "data_importer_query": "SQL query to execute for data importation.",
    "data_importer_save_importation": "Boolean value to determine if the imported data should be saved to a file.",
    "data_importer_file_name": "Name of the file where the imported data should be saved.",
    "prepare_dataframe_from_db_cols_for_query" : "List of columns to be queried from the database.",
    "preprocess_time_series_data_resample_freq": "Frequency for resampling the time series data.",
    "preprocess_time_series_data_aggregation_func": "Aggregation function to use when resampling.",
    "preprocess_time_series_data_method":"Method to use for interpolating missing values.",
    "preprocess_time_series_data_outlier_cols": "Columns in which to detect and handle outliers.",
    "ini_train": "Start time for the training data period.",
    "fin_train": "End time for the training data period.",
    "fin_test": "End time for the testing data period.",
    "name_time_column": "Name of the time column in the data.",
    "name_id_sensor_column": "Name of the sensor ID column in the data.",
    "id_device": "ID of the device from which data is collected.",
    "names_objective_variable": "Name of the objective variable to predict.",
    "predictor": "Name of the machine learning model to use for prediction.",
    "X_name_features": "List of feature names for the predictor; if null, all columns except specified are used.",
    "Y_name_features": "Name of the target feature for prediction.",
    "n_lags": "Number of lag values to include as features.",
    "n_predictions": "Number of future time steps to predict.",
    "lag_columns": "List of columns for which lag values should be created; if null, specified columns are used.",
    "lead_columns": "Name of the column for lead values (i.e., future values to predict).",
    "num_obs_to_predict": "Number of observations to predict; if null, all observations are predicted.",
    "scale_in_preprocessing": "Boolean value to determine if scaling should be applied during preprocessing.",
    "path_to_save_model": "Path where the trained machine learning model should be saved.",
    "save_preprocessing": "Boolean value to determine if preprocessing steps should be saved.",
    "folder_name_model": "Name of the folder where the machine learning model should be saved.",
    "folder_name_preprocessed_data": "Name of the folder where preprocessed data should be saved.",
    "folder_name_time_execution": "Name of the folder to save execution time information; if null, a timestamped folder is created.",
    "machine_learning_model_args": "Dictionary of additional arguments to pass to the machine learning model."
}

# Initialisation of a instance of ConfigManager
config_manager = ConfigManager("../config")

# Save sklearn details (version and date of doc scrapping)
config_manager.save_config(
    config_filename = "sklearn_details", 
    config = sklearn_details, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save parameters allowed in each regressor
config_manager.save_config(
    config_filename = "all_regressors_with_its_parameters", 
    config = all_regressors_with_its_parameters, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save all info in each regressor including explanation of each regressor and its parameters 
config_manager.save_config(
    config_filename = "all_sklearn_regressors_with_all_info", 
    config = all_regressors_with_all_info, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save initial_dict_parameters metadata
config_manager.save_config(
    config_filename = "legible_names_of_own_parameters", 
    config = initial_dict_parameters_legible_names, 
    subfolder = "models_parameters/metadata",
    create = True
)
config_manager.save_config(
    config_filename = "descriptions_of_own_parameters", 
    config = initial_dict_parameters_descriptions, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save all config
[config_manager.save_config(regressor, initial_dict_parameters, subfolder = "models_parameters") 
 for regressor in all_regressors]

# Update predictor parameter according to the regressor
# [config_manager.update_config(regressor, {"predictor":regressor}, subfolder = "models_parameters") 
#  for regressor in all_regressors]

# 1- Update predictor parameter according to the regressor
# 2- Add own parameters of each regressor

for regressor, parameters in all_regressors_with_its_parameters.items():
    new_values = {
        "predictor": regressor,
        "regressor_params": {
            param_info['parameter']: evaluate_or_return_default(param_info['value_default'])
            for param_info in all_regressors_with_all_info[regressor]['parameters_info']
            # Loop only parameters with a default values and valid parameters according get_all_regressors_with_its_parameters() function
            if param_info['value_default'] is not None and param_info['parameter'] in parameters 
        }
    }
    config_manager.update_config(
        config_filename = regressor, 
        new_values = new_values,
        subfolder="models_parameters"
    )