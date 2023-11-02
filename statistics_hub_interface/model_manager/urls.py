from django.urls import path
from . import views

urlpatterns = [
    path('', views.train_model, name='train_model'),
    path('get_model_parameters/<str:model_type>/', views.get_model_parameters, name='get_model_parameters')
]