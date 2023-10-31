import os
from predictions import process_model_machine_learning

path_to_save_model = os.path.join("..","models")
folder_name_model = "KNeighborsRegressor"
folder_name_range_train = "initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0"
folder_name_time_execution = "execution-time-2023_10_26_13_31_40"
folder_name_preprocessed_data = "preprocessed-data-to-use-in-model"
save = True
folder_name_predictions = "predictions"
name_column_objective_variable_of_scaler = "y"
flag_predictions_done = "predictions-done.txt"
overwrite_predictions = False

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