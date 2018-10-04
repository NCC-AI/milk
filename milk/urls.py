from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

app_name = 'milk'

urlpatterns = [
    path('', views.TrainView.as_view(), name='train'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('paint/', views.PaintView.as_view(), name='paint'),
]
