from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

app_name = 'milk'

urlpatterns = [
    path('', views.TrainView.as_view(), name='train'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('home/', views.HomeView.as_view(), name='home'),
    path('progress/<int:pk>/', views.progress, name='progress'),
]
