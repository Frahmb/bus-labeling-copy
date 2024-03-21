from django.urls import path
from .views import download


app_name = 'filesharing'
urlpatterns = [
    path('download/<file_id>',
         download, name='download'),
]
