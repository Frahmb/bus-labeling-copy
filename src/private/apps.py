from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class PrivateConfig(AppConfig):
    name = 'private'
    verbose_name = _('Private Dataset')
