import os
from predictions import evaluate_model


path_to_save_model = os.path.join("..","models")
folder_name_model = "KNeighborsRegressor"
folder_name_range_train = "initrain-2023_4_18_0_0_0-UTC0___fintrain-2023_4_25_0_0_0-UTC0"
folder_name_time_execution = "execution-time-2023_10_26_13_31_40" 
folder_name_predictions = "predictions"
save = True
flag_name = "evaluations-done"
flag_content = ""
flag_subfolder = None

# summarise_mae functions arguments
freq = None 
group_by_timestamp = True

evaluate_model(
    path_to_save_model= path_to_save_model,
    folder_name_model=folder_name_model,
    folder_name_range_train=folder_name_range_train,
    folder_name_time_execution=folder_name_time_execution,
    folder_name_predictions = folder_name_predictions,
    save=save,
    flag_name=flag_name,
    flag_content=flag_content,
    flag_subfolder=flag_subfolder,
    freq = freq,
    group_by_timestamp = group_by_timestamp
)