import os
import zipfile
import json
import torch
import PIL
from PIL import Image
from io import BytesIO
from django.http import HttpResponse
from django.conf import settings
from django.utils.translation import gettext_lazy as _
import pandas as pd
from sklearn.model_selection import train_test_split
from .models import (
    BUSDataset
)
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .MAE_Tools.pos_embed import interpolate_pos_embed
from .MAE_Tools import misc

with open(os.path.join(os.path.dirname(__file__), 'Label file README.txt')) as f:
    README = f.read()


def export_case(queryset, user=None):
    records = [i.export() for i in queryset.all()]
    image_files = []
    for record in records:
        for image in record['images']:
            image_files.append(image['image'])
            if 'mask' in image:
                image_files.append(image['mask'])

    zf_str = BytesIO()
    zf = zipfile.ZipFile(zf_str, 'w', zipfile.ZIP_DEFLATED)
    for i in image_files:
        zf.write(os.path.join(settings.MEDIA_ROOT, i), i)
    zf.writestr('label.json', json.dumps(records, indent=4))
    zf.writestr('README.txt', README)
    zf.close()

    zf_str.seek(0)
    response = HttpResponse(zf_str.getvalue(),
                            content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename="download.zip"'
    return response


def export_self_labeled(modeladmin, request, queryset):
    return export_case(queryset, request.user)



def split_tr_v_t(df_input, frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''


    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
        (frac_train, frac_val, frac_test))
    '''
    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.
    '''
    # Split original dataframe into train and temp dataframes.
    df_train, df_temp = train_test_split(df_input, test_size=(1.0 - frac_train), random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = train_test_split(df_temp, test_size=relative_frac_test, random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

def build_dataset(data_id):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # eval transform
    bus_dataset = BUSDataset.objects.get(name=data_id)
    bus_dataset.load_dataset()

    crop_pct = 224 / 256
    size = int(224 / crop_pct)
    t = []
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(224))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    transform = transforms.Compose(t)

    root = os.path.join(bus_dataset.path, 'images')
    newroot = os.path.join(settings.MEDIA_ROOT, root)

    dataset = datasets.ImageFolder(newroot, transform=transform)

    return dataset


def predict(data_loader, model, device):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_predictions = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)

        _, predicted_labels = torch.max(output, 1)
        all_predictions.extend(predicted_labels.cpu().numpy())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return all_predictions






def export_all_labeled(modeladmin, request, queryset):
    return export_case(queryset)


export_all_labeled.short_description = _('Export all labels')
