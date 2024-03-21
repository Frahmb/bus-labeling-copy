"""MIDA URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from labeling.admin import labeling_site
from labeling.urls import urlpatterns as labeling_url
from user.urls import urlpatterns as user_url
from public.urls import urlpatterns as public_url
from auxiliary.urls import urlpatterns as auxiliary_url


urlpatterns = [
    path('admin/', admin.site.urls),
    path('labeling/', labeling_site.urls),
    path('i18n/', include('django.conf.urls.i18n')),
    path('filesharing/', include('filesharing.urls')),
] + labeling_url + public_url + user_url + auxiliary_url

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
