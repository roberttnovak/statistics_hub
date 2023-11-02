from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import render
import sys
sys.path.append(str(Path('../src')))
from ConfigManager import ConfigManager
from django.shortcuts import render


def get_model_parameters(request, model_type):
    config_manager = ConfigManager("../config")
    model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters")
    return JsonResponse(model_parameters_default)

def train_model(request):
    model_type = ""
    message = ""
    model_parameters_default = None  # inicializa model_parameters_default a None

    config_manager = ConfigManager("../config")

    if request.method == 'POST':
        model_type = request.POST['model_type']
        model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters")  # mueve esta línea aquí

    
    models_list = config_manager.list_configs(subfolder="models_parameters")
    all_regresors_with_its_parameters = config_manager.load_config("all_regresors_with_its_parameters", subfolder="models_parameters/metadata")
    sklearn_parameters = all_regresors_with_its_parameters.get(model_type,{}) if model_type else {}  # verifica si model_type está definido antes de intentar acceder a all_regresors_with_its_parameters

    return render(
        request, 
        'model_manager/train_model.html', 
        {
            'models_list': models_list, 
            "model_parameters_default": model_parameters_default
        }
    )

