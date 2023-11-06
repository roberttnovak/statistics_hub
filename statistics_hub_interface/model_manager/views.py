import json
import os
from pathlib import Path
from django.http import JsonResponse
from django.shortcuts import redirect, render
import sys
from django.shortcuts import render

sys.path.append(str(Path("../src")))
from sql_utils import test_database_connection
from own_utils import modify_json_values
from ConfigManager import ConfigManager
from django.contrib.auth import authenticate, login

creds_path = '../global_creds/sql.json'

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
            return redirect('data_source_selection')  # Asegúrate de reemplazar 'home' con el nombre de tu vista de inicio
        else:
            # Devolver al formulario de login con un mensaje de error
            return render(request, 'model_manager/login.html', {'form': { 'errors': True }})
    else:
        return render(request, 'model_manager/login.html')