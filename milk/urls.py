from django.urls import path
from . import views

app_name = 'milk'

urlpatterns = [
    path('', views.TrainView.as_view(), name='train'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('paint/', views.PaintView.as_view(), name='paint'),
]
