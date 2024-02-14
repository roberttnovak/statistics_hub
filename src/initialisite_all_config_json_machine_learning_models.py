"""
This script initializes configurations for scikit-learn regression models, performing operations such as:

1. Retrieving a list of all scikit-learn regressors along with their parameters and additional information obtained
    from the official documentation scraped from the scikit-learn website.
2. Constructing a template dictionary for initial parameter configurations and its data type.
3. Creating legible names and descriptions for each parameter in the template dictionary.
4. Creating a mapping of wrong data types for parameters obtained from the scikit-learn documentation to fix data type.
5. Saving the list of regressors with their parameters as a configuration file.
6. Saving the list of regressors with their parameters and additional information as configuration files.
"""
import sklearn 
import datetime
from own_utils import evaluate_or_return_default, update_deep_nested_dict_value
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
    'regressor_params':{},
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
    'save_args': 
    {
        'scale_in_preprocessing': True, 
        'save_preprocessing': True, 
        'folder_name_model': None, 
        'folder_name_time_execution': None 
    }
}

data_type_dict_parameters_types = {
    'data_importer_args': {
        'data_importer_automatic_importation': 'bool', 
        'data_importer_database': 'str',
        'data_importer_query': 'str',  
        'data_importer_save_importation': 'bool', 
        'data_importer_file_name': 'str'  
    },
    'preprocess_time_series_data_args': {
        'preprocess_time_series_data_resample_freq': 'str', 
        'preprocess_time_series_data_aggregation_func': 'str', 
        'preprocess_time_series_data_method': 'str', 
        'preprocess_time_series_data_outlier_cols': 'list' 
    },
    'split_train_test_args': {
        'ini_train': 'str',
        'fin_train': 'str',
        'fin_test': 'str'
    },
    'regressor_params': 'dict',
    'time_serie_args': {
        'name_time_column': 'str',
        'name_id_sensor_column': 'str',
        'id_device': 'str', 
        'names_objective_variable': 'str', 
        'prepare_dataframe_from_db_cols_for_query': 'list', 
        'X_name_features': 'list',  
        'Y_name_features': 'str',
        'n_lags': 'int',
        'n_predictions': 'int',
        'lag_columns': 'list',  
        'lead_columns': 'str',
        'num_obs_to_predict': 'int'  
    },
    'save_args': {
        'scale_in_preprocessing': 'bool', 
        'save_preprocessing': 'bool', 
        'folder_name_model': 'str',  
        'folder_name_time_execution': 'str'  
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

#TODO: Pensar en una forma mejor de manejar todos y cada uno de estos tipos de datos que se han hecho en el scrapping
# Por ejemplo, en aquellos donde se puede meter un str o un float
mapping_wrong_data_type_sklearn = {
    "object": "str",
    "{linear, square, exponential}" : "str",
    "int, RandomState instance or None": "int",
    "int or float": "float",
    "{squared_error, friedman_mse, absolute_error, poisson}": "srt",
    "{best, random}": "str",
    "int, float or {sqrt, log2}": "float",
    "non-negative float": "float",
    "{mean, median, quantile, constant}": "str",
    "int or float or array-like of shape (n_outputs,)": "int",
    "float in [0.0, 1.0]": "float",
    "bool or array-like of shape (n_features, n_features), default=False": "bool",
    "int, RandomState instance": "int",
    "{cyclic, random}": "str",
    "float or list of float": "float",
    "array-like": "list",
    "auto, bool or array-like of shape (n_features, n_features)": "bool",
    "int, cross-validation generator or iterable": "int",
    "bool or int": "bool",
    "{squared_error, friedman_mse, absolute_error, poisson}, default=squared_error": "str",
    "{random, best}": "str",
    "int, float, {sqrt, log2} or None": "float",
    "{squared_error, absolute_error, friedman_mse, poisson}, default=squared_error": "str",
    "{sqrt, log2, None}, int or float": "float",
    "bool or callable": "bool",
    "{lbfgs, newton-cholesky}": "str",
    "kernel instance": "str",
    "float or ndarray of shape (n_samples,)": "float", 
    "fmin_l_bfgs_b, callable or None": "str",
    "{squared_error, absolute_error, huber, quantile}, default=squared_error": "str",
    "{friedman_mse, squared_error}": "str",
    "int or None": "int",
    "estimator or zero": "str",
    "{sqrt, log2}, int or float": "float",
    "{squared_error, absolute_error, gamma, poisson, quantile}, default=squared_error": "str",
    "array-like of {bool, int, str} of shape (n_features) or shape (n_categorical_features,)": "bool",
    "array-like of int of shape (n_features) or dict": "list",
    "{pairwise, no_interactions} or sequence of lists/tuples/sets of int": "str",
    "auto or bool": "str",
    "str or callable or None": "str",
    "int or float or None": "float",
    "{nan, clip, raise}": "str", 
    "bool or auto": "bool",
    "{uniform, distance}, callable or None": "str",
    "{auto, ball_tree, kd_tree, brute}": "str",
    "str, DistanceMetric object or callable": "str",
    "float or array-like of shape (n_targets,)": "float",
    "str or callable": "str",
    "bool, auto or array-like": "bool",
    "int, cross-validation generator or an iterable": "int", 
    "{aic, bic}": "str",
    "{epsilon_insensitive, squared_epsilon_insensitive}, default=epsilon_insensitive": "str",
    "array-like of shape(n_layers - 2,)": "list",
    "{identity, logistic, tanh, relu}": "str",
    "{lbfgs, sgd, adam}": "str",
    "{constant, invscaling, adaptive}": "str",
    "{linear, poly, rbf, sigmoid, precomputed} or callable, default=rbf": "str", 
    "{scale, auto} or float" : "str",
    "{nipals, svd}": "str",
    "{highs-ds, highs-ipm, highs, interior-point, revised simplex}": "str",
    "int (>= 1) or float ([0, 1])": "float",
    "callable": "str",
    "float in range [0, 1]": "float",
    "str, callable": "str",
    "{float, ndarray of shape (n_targets,)}": "float",
    "{auto, svd, cholesky, lsqr, sparse_cg, sag, saga, lbfgs}": "str",
    "array-like of shape (n_alphas,)": "list",
    "{auto, svd, eigen}": "str",
    "{l2, l1, elasticnet, None}": "str", 
    "float or None": "float", 
    "function": "str",
    "{auto, identity, log}": "str"
}

all_sklearn_regressors_set_parameters = {regressor: initial_dict_parameters.copy() for regressor in all_regressors}


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

# Save descriptions of own parameters (frontend purposes)
config_manager.save_config(
    config_filename = "descriptions_of_own_parameters", 
    config = initial_dict_parameters_descriptions, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save data type of own parameters
config_manager.save_config(
    config_filename = "data_type_of_own_parameters", 
    config = data_type_dict_parameters_types, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Save mapping of wrong data types (because of scrapping) in sklearn
config_manager.save_config(
    config_filename = "mapping_wrong_data_type_sklearn", 
    config = mapping_wrong_data_type_sklearn, 
    subfolder = "models_parameters/metadata",
    create = True
)

# Get parameters of each sklearn regressor
for regressor, metadata in all_regressors_with_all_info.items():
    regressor_params = {}
    for metadata_parameter in metadata['parameters_info']:
        parameter = metadata_parameter['parameter']
        value_default = evaluate_or_return_default(metadata_parameter['value_default'])
        regressor_params[parameter] = value_default
    all_sklearn_regressors_set_parameters[regressor]['regressor_params'] = regressor_params.copy()

# Save json in which will be saved parameters which user will change in application
config_manager.save_config(
    config_filename = "parameters", 
    config = all_sklearn_regressors_set_parameters, 
    subfolder = "models_parameters",
    create = True
)

#TODO: En algún momento, unificar todo el archivo que hay un un único diccionario. Por ejemplo:
# {
# 'data_importer_args': {
#     'data_importer_automatic_importation': {'value': False, 'type': bool, ...},
#     'data_importer_database': {'value': None, 'type': str, ...},
#     'data_importer_query': {'value': None, 'type': str, ...},
#     'data_importer_save_importation': {'value': True, 'type': bool, ...},
#     'data_importer_file_name': {'value': None, 'type': str, ...}
# },...
