from django.urls import path
from .views import CompetitionView
from django.views.generic.base import RedirectView


urlpatterns = [
    path('competition/<int:index>',
         CompetitionView.as_view(), name='competition'),
]
