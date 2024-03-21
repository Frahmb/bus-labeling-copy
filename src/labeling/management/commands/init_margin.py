import os
import pandas as pd
from django.core.management.base import BaseCommand
from labeling.models import BIRADSMarginChoice


class Command(BaseCommand):
    help = "Initialize BIRDAS Margin Choice"

    def handle(self, *args, **options):
        BIRADSMarginChoice.init_db()
