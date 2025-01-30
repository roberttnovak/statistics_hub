import numpy as np 
import pandas as pd
from ConfigManager import ConfigManager
from cleaning import prepare_dataframe_from_db
from predictions import create_cv_hyperparameters_all_sklearn_models, create_cv_hyperparameters_model, generate_cv_grid_regressor_sklearn, recommend_distribution_and_range

config_manager = ConfigManager("../config")

all_regressors_with_its_parameters_domains_and_gridsearchcv = config_manager.load_config(
    "models_parameters/metadata/all_regressors_with_its_parameters_domains_and_gridsearchcv"
)

gridsearchcv_hyperparameters_preprocessing = config_manager.load_config(
    "models_parameters/metadata/gridsearchcv_hyperparameters_preprocessing"
)

gridsearchcv_hyperparameters_model_space = config_manager.load_config(
    "models_parameters/metadata/gridsearchcv_hyperparameters_model_space"
)

df = pd.read_csv(r'..\data\instants_data_saved\2023-07-04_12-09-22.csv')
df = df.query("id_device == 'DBEM003'").reset_index(drop=True)
prepare_dataframe_from_db_cols_for_query = [
    "00-eco2",
    "00-temp",
    "01-hum",
    "01-tvoc",
    "02-pres",
    "03-siaq",
    "04-diaq"
]
# Prepare dataframe with selected columns
df = prepare_dataframe_from_db(
    df=df,
    cols_for_query=prepare_dataframe_from_db_cols_for_query,
)

#TODO: Terminar esto
results = create_cv_hyperparameters_all_sklearn_models(
    df=df,
    all_regressors_with_its_parameters_domains_and_gridsearchcv=all_regressors_with_its_parameters_domains_and_gridsearchcv,
    hyperparameters_preprocessing=gridsearchcv_hyperparameters_preprocessing,
    hyperparameters_model_space=gridsearchcv_hyperparameters_model_space
)
    
 

print()