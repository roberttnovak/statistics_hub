from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import redirect, render
import sys
sys.path.append(str(Path("../src")))
from ConfigManager import ConfigManager
from django.shortcuts import render


def get_model_parameters(request, model_type):
    config_manager = ConfigManager("../config")
    model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters")
    legible_names_of_parameters = config_manager.load_config("legible_names_of_parameters", subfolder="models_parameters/metadata")
    descriptions_of_parameters = config_manager.load_config("description_of_parameters", subfolder="models_parameters/metadata")
    response_data = {
        "model_parameters_default": model_parameters_default,
        "legible_names_of_parameters": legible_names_of_parameters,
        "descriptions_of_parameters": descriptions_of_parameters
    }
    return JsonResponse(response_data)


def train_model(request):
    model_type = ""
    model_parameters_default = None 
    legible_names_of_parameters = None    
    descriptions_of_parameters = None
    all_regresors_with_its_parameters = None    

    config_manager = ConfigManager("../config")

    if request.method == "POST":
        model_type = request.POST["model_type"]
        model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters") 
        legible_names_of_parameters = config_manager.load_config("legible_names_of_parameters", subfolder="models_parameters/metadata")
        descriptions_of_parameters = config_manager.load_config("description_of_parameters", subfolder="models_parameters/metadata")
        all_regresors_with_its_parameters = config_manager.load_config("all_regresors_with_its_parameters", subfolder="models_parameters/metadata")

    
    models_list = config_manager.list_configs(subfolder="models_parameters")

    sklearn_parameters = all_regresors_with_its_parameters.get(model_type,{}) if model_type else {}  # verifica si model_type está definido antes de intentar acceder a all_regresors_with_its_parameters

    return render(
        request, 
        "model_manager/train_model.html", 
        {
            "models_list": models_list, 
            "model_parameters_default": model_parameters_default,
            "legible_names_of_parameters" : legible_names_of_parameters,
            "descriptions_of_parameters" : descriptions_of_parameters
        }
    )

def data_source_selection(request):
    return render(request, 'model_manager/data_source_selection.html')

def upload_file(request):
    # Lógica para manejar la carga del CSV
    return render(request, 'model_manager/upload_file.html')

def connect_to_database(request):
    # Lógica para manejar la conexión a la base de datos
    return render(request, 'model_manager/connect_to_database.html')
