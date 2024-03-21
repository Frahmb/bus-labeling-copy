from django.urls import path
from django.views.generic.base import RedirectView
from .views import MaskingView


urlpatterns = [
    path('', RedirectView.as_view(url='/labeling')),
    path('masking/<masking_type>/<ds_name>', MaskingView.as_view(),
         name='masking'),
    path('masking/<masking_type>/<ds_name>/<int:obj_id>',
         MaskingView.as_view(), name='masking_change'),
]
