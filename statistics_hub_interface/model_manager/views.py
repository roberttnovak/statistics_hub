import json
import logging
import os
from pathlib import Path
from django.conf import settings
from django.http import HttpResponse, HttpResponseNotAllowed, HttpResponseRedirect, JsonResponse
from django.shortcuts import redirect, render
import sys
from django.shortcuts import render
from django.urls import reverse
from django.contrib.auth import authenticate, login
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
from django.contrib import messages
 
sys.path.append(str(Path("../src")))
from PersistanceManager import PersistenceManager
from sql_utils import test_database_connection
from own_utils import modify_json_values
from ConfigManager import ConfigManager
from predictions import run_time_series_prediction_pipeline


creds_path = '../global_creds/sql.json'

# ToDo: remove train_model
# ToDo: Use ConfigManager in all functions when necessary (for example upload_file instead manual modification)
# ToDo: Document better and apply good practice


@login_required
def user_resources(request):
    # Aquí puedes añadir lógica para obtener cualquier dato necesario para la plantilla
    # Por ejemplo, listas de modelos, conjuntos de datos, etc.
    return render(request, 'model_manager/user_resources.html')

@login_required
def user_resources_models(request):
    user = request.user
    pm = PersistenceManager(base_path=f"tenants/{user}/models")

    # Lista de modelos y sus rangos de entrenamiento.
    models_list = pm.get_available_models()
    
    # Diccionario para almacenar modelos, rangos y tiempos de ejecución.
    models_with_details = {}

    for model in models_list:
        ranges = pm.get_training_ranges_for_model(model)
        models_with_details[model] = {}

        for range_ in ranges:
            execution_times = pm.get_execution_times_for_model_and_range(model, range_)
            models_with_details[model][range_] = execution_times

    return render(request, 'model_manager/user_resources_models.html', {
        'models_with_details': models_with_details
    })

@login_required
def model_evaluation_time_execution(request, execution_time):
    # Lógica para manejar la vista de tiempo de ejecución
    return render(request, 'model_manager/model_evaluation_time_execution.html')

@login_required
def model_evaluation_train_range(request, training_range):
    # Lógica para manejar la vista del rango de entrenamiento
    return render(request, 'model_manager/model_evaluation_train_range.html')

@login_required
def model_evaluation_model(request, model):
    # Lógica para manejar la vista del modelo
    return render(request, 'model_manager/model_evaluation_model.html')

@login_required
def model_evaluation_all_models(request):
    # Lógica para manejar la vista del modelo
    return render(request, 'model_manager/model_evaluation_all_models.html')

@login_required
def get_models_list(request):
    user = request.user
    config_manager = ConfigManager(f"tenants/{user}/config")
    models_list = config_manager.list_configs(subfolder="models_parameters")
    return models_list

@login_required
def get_model_parameters(request, model_type):
    user = request.user
    config_manager = ConfigManager(f"tenants/{user}/config")
    model_parameters_default = config_manager.load_config(model_type, subfolder="models_parameters")
    legible_names_of_parameters = config_manager.load_config("legible_names_of_parameters", subfolder="models_parameters/metadata")
    descriptions_of_parameters = config_manager.load_config("description_of_parameters", subfolder="models_parameters/metadata")
    response_data = {
        "model_parameters_default": model_parameters_default,
        "legible_names_of_parameters": legible_names_of_parameters,
        "descriptions_of_parameters": descriptions_of_parameters
    }
    return response_data


@login_required
def model_train(request, model_name):
    if request.method == 'POST':
        config_path = f"tenants/{request.user.username}/config"
        config_log_filename = "config_logs"

        try:
            run_time_series_prediction_pipeline(config_path, model_name, config_log_filename)
            messages.success(request, "Model trained succesfully")
        except Exception as e:
            messages.error(request, f"Error during training: {e}")

        return redirect('model_selection') 

    else:
        return redirect('model_selection')

@login_required
def model_selection(request):
    selected_model = request.GET.get('selected_model')

    if request.method == 'POST':
        selected_model = request.POST.get('model_type')
        action = request.POST.get('action')
        if action == 'show_parameters':
            return redirect('model_parameters', model_name=selected_model)
        elif action == 'train_model':
            config_path = f"tenants/{request.user.username}/config"
            #ToDo: Tratar en algún momento los logs 
            config_log_filename = None
            try:
                run_time_series_prediction_pipeline(config_path, selected_model, config_log_filename)
                messages.success(request, "Model trained succesfully")
            except Exception as e:
                messages.error(request, f"Error during training: {e}")

            return redirect('model_train', model_name=selected_model)

    models_list = get_models_list(request)
    return render(request, 'model_manager/model_selection.html', {
        'models_list': models_list,
        'selected_model': selected_model  
    })


@login_required
def model_parameters(request, model_name):
    """
    View for handling the display and update of model parameters.

    Parameters:
    - request: HttpRequest object containing metadata and form data for POST requests.
    - model_name: str, the name of the model whose parameters are being edited.

    Returns:
    - HttpResponse: Rendered HTML page for GET requests or a redirect for POST requests.
    """
    user = request.user

    # Instantiate the ConfigManager with the path to the user's tenant configuration.
    config_manager = ConfigManager(f"tenants/{user}/config")

    if request.method == 'POST':
        if not model_name:
            model_name = request.POST.get('model_type')  
        # Capture the form data as a dictionary.
        updated_parameters = {param: request.POST.get(param) for param in request.POST if param != 'csrfmiddlewaretoken'}

        try:
            # Update the model's configuration using the captured form data.
            config_manager.update_config(model_name, updated_parameters, subfolder="models_parameters")

            # Redirect to avoid post data resubmission if the user refreshes the page.
            # return redirect('../../model_selection', model_name=model_name)
            return HttpResponseRedirect(f"{reverse('model_selection')}?selected_model={model_name}")

        except FileNotFoundError as e:
            # Log the error and handle it as appropriate.
            # (Not shown: You should include logging here for the error e)
            messages.error(request, "An error occurred while updating the parameters. Please try again.")

    # For a GET request, load the existing parameters to display in the form.
    try:
        # Load the model's current parameters from the configuration.
        model_parameters = config_manager.load_config(model_name, subfolder="models_parameters")

    except FileNotFoundError:
        # If the config file is not found, handle the error appropriately (e.g., log it, send an error message to the template, etc.)
        model_parameters = {}

    # Render the page with the current model parameters.
    return render(request, 'model_manager/model_parameters.html', {
        'model_name': model_name,
        'model_parameters': model_parameters
    })

def data_source_selection(request):
    return render(request, 'model_manager/data_source_selection.html')

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
    
@login_required  # Ensures that only authenticated users can access this view.
def upload_file(request):
    """
    Handles file uploads by authenticated users, displaying an upload form for GET requests
    and processing the file for POST requests. After a successful file upload, the user is
    redirected to avoid duplicate submissions if the page is refreshed. If a file is selected
    for training, 'data_source.json' is created or updated with the selected file's name,
    and the user is redirected to the '' view.

    Parameters:
    - request: HttpRequest object containing metadata, the uploaded file, or the selected file name.

    Returns:
    - HttpResponse: Redirect to the upload page to display the form and list of uploaded files,
                    redirect to the train model page after selecting a file, or a 405 response for methods not allowed.
    """
    if request.method == 'POST':
        if 'local_file' in request.FILES:
            # Handle file upload on POST request.
            user = request.user

            # Define the path to the 'data' directory within the user's tenant directory.
            tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', user.username, 'data')
            
            # Create the directory if it doesn't exist.
            os.makedirs(tenant_data_dir, exist_ok=True)

            # Use FileSystemStorage with the path to the tenant's 'data' directory.
            fs = FileSystemStorage(location=tenant_data_dir)

            # Get the uploaded file from the request.
            myfile = request.FILES['local_file']

            # Check for directory traversal in filename.
            if '..' in myfile.name or '/' in myfile.name:
                raise ValueError("Invalid filename.")

            try:
                # Save the file.
                filename = fs.save(myfile.name, myfile)
                
                # Get the URL of the saved file.
                uploaded_file_url = fs.url(filename)

                # Redirect to the same view to show the form and file list.
                return HttpResponseRedirect(reverse('upload_file'))
            except Exception as e:
                # Handle any unexpected exceptions during file save operation.
                # Log the exception and return an appropriate HTTP response.
                # (Logging not shown here, but should be implemented.)
                return HttpResponse(str(e), status=500)
        
        elif 'selected_file' in request.POST:
            # File selection logic for training
            selected_file = request.POST['selected_file']
            if '..' in selected_file or '/' in selected_file:
                # Additional security check
                raise ValueError("Invalid filename.")
            tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', request.user.username)
            data_source_path = os.path.join(tenant_data_dir, 'data_source.json')
            os.makedirs(tenant_data_dir, exist_ok=True)  # Ensure the directory exists
            with open(data_source_path, 'w') as file:
                # Write the selected file name to 'data_source.json'
                json.dump({'selected_file': selected_file}, file)
            return redirect('model_selection')  # Redirect to the training view
        else:
            # Invalid POST request
            return HttpResponse("Invalid POST request.", status=400)

    elif request.method == 'GET':
        # Show the upload form on GET request.
        user = request.user

        # Define the path to the 'data' directory within the user's tenant directory.
        tenant_data_dir = os.path.join(settings.BASE_DIR, 'tenants', user.username, 'data')

        # Use FileSystemStorage to list files in the directory.
        fs = FileSystemStorage(location=tenant_data_dir)

        # List files already uploaded.
        user_files = [{'name': file, 'url': fs.url(file)} for file in fs.listdir('')[1]]

        # Render the upload form and list of files.
        return render(request, 'model_manager/upload_file.html', {
            'user_files': user_files
        })

    else:
        # Return a response indicating that the method is not allowed.
        # Only POST and GET are supported.
        return HttpResponseNotAllowed(['POST', 'GET'])

