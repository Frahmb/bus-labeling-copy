# Generated by Django 3.1 on 2023-06-22 23:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auxiliary', '0002_auto_20230622_1556'),
    ]

    operations = [
        migrations.AddField(
            model_name='competition',
            name='submit_file',
            field=models.BooleanField(default=False, verbose_name='Submit File'),
        ),
    ]