import sys
from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class LabelingConfig(AppConfig):
    name = 'labeling'
    verbose_name = _('My Dataset')

    def ready(self):
        if 'runserver' in sys.argv:
            from .models import BUSDataset
            for ds in BUSDataset.objects.all():
                ds.register_admin_model()
            return True
