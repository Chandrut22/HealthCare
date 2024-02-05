from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='index'),
    path('diabetes_prediction/',views.diabetes_prediction,name='diabetes_prediction'),
    path('disease_prediction/',views.disease_prediction,name='disease_prediction'),
    path('heart_attack_prediction/',views.heart_attack_prediction,name='heart_attack_prediction'),
    path('liver_disease_prediction/',views.liver_disease_prediction,name='liver_disease_prediction'),

    path('process_columns/', views.process_columns, name='process_columns'),
    
]