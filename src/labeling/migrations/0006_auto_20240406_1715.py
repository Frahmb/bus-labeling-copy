# Third times the charm...

from django.db import migrations


def create_initial_checkpoint(apps, schema_editor):
    ModelCheckpoint = apps.get_model('labeling', 'ModelCheckpoint')
    ModelCheckpoint.objects.create(
        model_name='MAE - MedAI Official',
        model_type='MAE',
        checkpoint_path='/mnt/c/BryanTestCase/bus-labeling-copy/src/MedAI-MAE/checkpoints/mae_BUS_checkpoint_gpf.pth'
    )


class Migration(migrations.Migration):

    dependencies = [
        ('labeling', '0005_auto_20240402_1537'),
    ]

    operations = [
        migrations.RunPython(create_initial_checkpoint),
    ]