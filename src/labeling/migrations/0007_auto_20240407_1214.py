# Fourth times the charm...

from django.db import migrations


def create_initial_checkpoint(apps, schema_editor):
    ModelCheckpoint = apps.get_model('labeling', 'ModelCheckpoint')
    ModelCheckpoint.objects.create(
        model_name='MAE - MedAI Official',
        model_type='MAE',
        checkpoint_name='mae_BUS_checkpoint_gpf.pth'
    )


class Migration(migrations.Migration):

    dependencies = [
        ('labeling', '0006_rename_checkpoint_path_modelcheckpoint_checkpoint_name'),
    ]

    operations = [
        migrations.RunPython(create_initial_checkpoint),

    ]
