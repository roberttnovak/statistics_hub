from django.urls import path
from . import views

urlpatterns = [
    path('train_model', views.train_model, name='train_model'),
    path('get_model_parameters/<str:model_type>/', views.get_model_parameters, name='get_model_parameters'),
    path('data_source_selection', views.data_source_selection, name='data_source_selection'),
    path('upload_file/', views.upload_file, name='upload_file'),
    path('connect_to_database/', views.connect_to_database, name='connect_to_database'),
    path('', views.user_login, name='login'),
]