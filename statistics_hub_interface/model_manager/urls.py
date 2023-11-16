from django.urls import path
from . import views

urlpatterns = [
    path('model_train/<str:model_name>/', views.model_train, name='model_train'),
    path('model_selection/', views.model_selection, name = 'model_selection'),
    path('model_parameters/', views.model_parameters, name = 'model_parameters'),
    path('get_model_parameters/<str:model_type>/', views.get_model_parameters, name='get_model_parameters'),
    path('model_parameters/<str:model_name>/', views.model_parameters, name='model_parameters'),
    path('user_resources/', views.user_resources, name='user_resources'),
    path('data_source_selection/', views.data_source_selection, name='data_source_selection'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('connect_to_database/', views.connect_to_database, name='connect_to_database'),
    path('', views.user_login, name='login'),
]