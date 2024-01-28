
import json
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save
from django.db.models.signals import pre_delete
from django.dispatch import receiver
import os
import shutil
from django.conf import settings

from pathlib import Path
import sys

sys.path.append(str(Path("../src")))
from ConfigManager import ConfigManager
from sklearn_utils import get_all_regressors

# ToDo: Investigar lo de si el instance.username puede dar problemas con respecto a .user que se usa en las otras funciones
# en las views de model_manager

class CustomUser(AbstractUser):
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name="customuser_set",
        related_query_name="customuser",
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="customuser_set",
        related_query_name="customuser",
    )

@receiver(post_save, sender=CustomUser)
def create_user_directory(sender, instance, created, **kwargs):
    """
    Upon creation of a new CustomUser instance, this function is triggered to
    create a dedicated tenant directory named after the username. It also
    copies the global configuration and credentials into the user's directory.
    
    Parameters:
    - sender (Model): The model class that sent the signal.
    - instance (CustomUser): The instance of the user that was saved.
    - created (bool): Flag that indicates whether a new record was created.
    - kwargs: Additional keyword arguments.
    """
    if created:
    
        tenant_dir = os.path.normpath(os.path.join(settings.BASE_DIR, 'tenants', instance.username))
        data_dir = os.path.normpath(os.path.join(tenant_dir, 'data'))  # Subdirectorio para archivos subidos
        os.makedirs(data_dir, exist_ok=True)
        
        # and copy the global configuration to the user's 'config' directory
        global_config_dir = os.path.abspath(os.path.join(settings.BASE_DIR,  '..', 'config'))
        user_config_dir = os.path.join(tenant_dir, 'config')
        shutil.copytree(global_config_dir, user_config_dir, dirs_exist_ok=True)

        # we copy them to a 'creds' folder within the tenant's directory.
        try:
            global_creds_dir = os.path.abspath(os.path.join(settings.BASE_DIR, '..', 'global_creds'))
            user_creds_dir = os.path.normpath(os.path.join(tenant_dir, 'creds'))
            shutil.copytree(global_creds_dir, user_creds_dir, dirs_exist_ok=True)
        except FileNotFoundError:
            # Create the creds directory and a default creds.json file.
            os.makedirs(user_creds_dir, exist_ok=True)
            default_creds = {
                "ssh_host": "",
                "ssh_port": None, 
                "ssh_user": "",
                "ssh_password": "",
                "db_server": "",
                "db_user": "",
                "db_password": ""
            }
            with open(os.path.join(user_creds_dir, 'creds.json'), 'w') as f:
                json.dump(default_creds, f, indent=4)
        # Change correct path to save models 
        config_manager = ConfigManager("../statistics_hub_interface/tenants/admin/config")
        
        all_regressors = get_all_regressors()

        # Update paths to save model
        # [config_manager.update_config(regressor, {"path_to_save_model": f"tenants/{instance.username}/models"}, subfolder = "models_parameters") 
        # for regressor in all_regressors]
        config_manager.update_config("common_parameters", 
                                     {
                                        "path_to_save_model": f"tenants/{instance.username}/models",
                                        "data_importer_creds_path": f"tenants/{instance.username}/creds",
                                        "data_importer_path_instants_data_saved": f"tenants/{instance.username}/data"
                                      }, 
                                     subfolder = "models_parameters/common_parameters")


@receiver(pre_delete, sender=CustomUser)  # Use pre_delete if you want to delete before the user is actually removed from the database
def delete_user_directory(sender, instance, **kwargs):
    """
    Before deleting a CustomUser instance, this function is triggered to
    remove the associated tenant directory along with its contents.
    
    Parameters:
    - sender (Model): The model class that sent the signal.
    - instance (CustomUser): The instance of the user that is about to be deleted.
    - kwargs: Additional keyword arguments.
    """
    tenant_dir = os.path.join(settings.BASE_DIR, 'tenants', instance.username)
    if os.path.exists(tenant_dir):
        shutil.rmtree(tenant_dir)  # Use rmtree to delete the directory tree

# ToDo: Crear una forma más automática para borrar usuarios. De momento:
# python manage.py shell
# ----------
# from users.models import CustomUser
# name = 'name'
# user_to_delete = CustomUser.objects.get(username=name)
# user_to_delete.delete()
# try:
#     user = CustomUser.objects.get(username=name)
#     print("El usuario aún existe.")
# except CustomUser.DoesNotExist:
#     print("El usuario ha sido eliminado.")
# ----------