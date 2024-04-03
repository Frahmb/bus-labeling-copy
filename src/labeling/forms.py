import os
import re
import uuid
import shutil
import zipfile
import pandas as pd
from django.utils.translation import gettext_lazy as _
from django import forms
from django.core.validators import FileExtensionValidator
from django.conf import settings
from .models import BUSDataset, ModelCheckpoint



class UploadingDatasetForm(forms.Form):
    name = forms.CharField(
        label=_('Dataset Name'), max_length=120, required=True)
    data_file = forms.FileField(label=_('Zipped Dataset File'),
                                validators=[FileExtensionValidator(['zip'])],
                                required=True)

    public_permission = forms.ChoiceField(label=_('Public Permission'),
                                          choices=BUSDataset.PERM)

    def clean_name(self):
        name = self.cleaned_data.get('name')
        if len(name) < 4:
            raise forms.ValidationError(
                _('Dataset name must be at lease 4 characters long'))

        if BUSDataset.objects.filter(name=name).count() > 0:
            raise forms.ValidationError(
                _('The name "%(name)" is already taken. Try picking a new one'),
                params={'name': name}
            )

        if not re.match(r'^\w[\w\d ]*$', name):
            raise forms.ValidationError(
                _('The name should only contains letters, numbers and spaces. '
                  'The first character must be letters.')
            )

        return name

    def clean(self):
        name = self.cleaned_data.get('name')
        if name is None:
            return
        uuid_str = str(uuid.uuid1()).replace('-', '')
        base_name = '_'.join([name, uuid_str])
        file_obj = self.cleaned_data.get('data_file')

        # store file on uploading location
        full_file_name = os.path.join(
            settings.UPLOADING_ROOT, base_name) + '.zip'
        with open(full_file_name, 'wb+') as des:
            for chunk in file_obj.chunks():
                des.write(chunk)

        # extracting data
        zf = zipfile.ZipFile(full_file_name)
        out_dir = os.path.join(settings.MEDIA_ROOT, base_name)
        zf.extractall(out_dir)
        # remove the zip file
        os.remove(full_file_name)

        # Examine file
        # the zipped file must contains label.csv
        label_file = os.path.join(out_dir, 'label.csv')
        if not os.path.exists(label_file):
            # clean extracted folder
            shutil.rmtree(out_dir)
            raise forms.ValidationError(
                {'data_file': _('The zipped file must includes label.csv')})

        df = pd.read_csv(label_file)

        # case_id column
        # must be integer
        # must >= 0
        if df.columns[0] != 'case_id':
            raise forms.ValidationError(
                {'data_file': _('The first column must be "case_id"')})
        if not pd.api.types.is_integer_dtype(df.dtypes[0]):
            raise forms.ValidationError(
                {'data_file': _('The case_id must be integer, '
                                'e.g. 1, 2, 3 ...')})
        if not (df['case_id'] >= 0).all():
            raise forms.ValidationError(
                {'data_file': _('The case_id must be 0 or positive '
                                'number (>= 0) instead of negative')})

        # image column
        # must be string
        if df.columns[1] != 'image':
            raise forms.ValidationError(
                {'data_file': _('The second column must be "image"')})
        if not pd.api.types.is_string_dtype(df.dtypes[1]):
            raise forms.ValidationError(
                {'data_file': _('The image column must be string type')})

        # tumor_type column
        # must be string
        # must be one of [N, B, M, O, U]
        if df.columns[2] != 'tumor_type':
            raise forms.ValidationError(
                {'data_file': _('The third column must be "tumor_type"')})
        if not pd.api.types.is_string_dtype(df.dtypes[1]):
            raise forms.ValidationError(
                {'data_file': _('The tumor_type column must be string type')})
        if not df['tumor_type'].str.upper().isin(
                ['N', 'B', 'M', 'O', 'U']).all():
            raise forms.ValidationError(
                {'data_file': _(
                    'The tumor_type must be one of following options: '
                    'N(normal), B(benign), M(malignant), O(others), U(unknown)'
                )})

        self.cleaned_data['data_file'] = base_name
        return self.cleaned_data

'''Class added by Jackson Baldwin for Senior Capstone Design - Spring 2024
Form used to split a dataset into training, validation, and test sets to be used by the retrain model feature
Code additionally found in admin.py under "splitting", and under split_dataset.html
'''
class SplittingDatasetForm(forms.Form):

    #selecting an ID for the newly split dataset
    name = forms.CharField(
        label=_('Select a Split Dataset ID'), max_length=120, required=True)

    public_permission = forms.ChoiceField(label=_('Public Permission'),
        choices=BUSDataset.PERM)

    # Add a new field for private databases
    private_database = forms.ModelChoiceField(
        #queryset=BUSDataset.objects.filter(permission = 'private'),
        queryset=BUSDataset.objects.all(),
        label='Select Dataset',
        empty_label="Choose a dataset",
        required=True, # Ensure this is set to True
    )


    training_percentage = forms.IntegerField(
        label='Training Percentage',
        min_value=0,
        max_value=100,
        required=True,
        help_text='Enter the percentage for the training set (0-100).'
    )
    validation_percentage = forms.IntegerField(
        label='Validation Percentage',
        min_value=0,
        max_value=100,
        required=True,
        help_text='Enter the percentage for the validation set (0-100).'
    )
    test_percentage = forms.IntegerField(
        label='Test Percentage',
        min_value=0,
        max_value=100,
        required=True,
        help_text='Enter the percentage for the test set (0-100).'
    )
    def clean_name(self):
        name = self.cleaned_data.get('name')
        if len(name) <  4:
            raise forms.ValidationError(
                _('Dataset name must be at least  4 characters long'))

        if BUSDataset.objects.filter(name=name).count() >  0:
            raise forms.ValidationError(
                _('The name "%(name)" is already taken. Try picking a new one'),
                params={'name': name}
            )   

        if not re.match(r'^\w[\w\d ]*$', name):
            raise forms.ValidationError(
                _('The name should only contains letters, numbers and spaces. '
                  'The first character must be letters.')
            )

        return name

    def clean(self):
        name = self.cleaned_data.get('name')
        if name is None:
            return
        cleaned_data = super().clean()
        training_percentage = cleaned_data.get('training_percentage')
        validation_percentage = cleaned_data.get('validation_percentage')
        test_percentage = cleaned_data.get('test_percentage')

        total_percentage = training_percentage + validation_percentage + test_percentage
        if total_percentage !=  100:
            raise forms.ValidationError('The sum of training, validation, and test percentages must equal  100%.')

        self.cleaned_data['data_file'] = name
        return self.cleaned_data



'''Class added by Jackson Baldwin for Senior Capstone Design - Spring 2024
Form used to retrain a model using the split datasets from the split_dataset form, as well as input selected by a user.
Code additionally found in admin.py under "retrain", and under retrain_model.html
'''
class ModelRetrainForm(forms.Form):
    
    datasets = forms.ModelChoiceField(
        queryset=BUSDataset.objects.all(),
        label='Select Dataset',
        empty_label="Choose a dataset"
    )
    epochs = forms.IntegerField(
        label='Number of Epochs',
        min_value=1,
        required=True
    )
    learning_rate = forms.FloatField(
        label='Learning Rate',
        min_value=0.0,
        required=True
    )
    batch_size = forms.FloatField(
        label='Batch Size',
        min_value=0.0,
        required=True
    )
    name = forms.CharField(
    label=_('Select a New Model ID'), max_length=120, required=True)

    def clean(self):
        cleaned_data = super().clean()

    

    def clean_name(self):
        name = self.cleaned_data.get('name')
        if len(name) <  4:
            raise forms.ValidationError(
                _('Dataset name must be at least  4 characters long'))

        if BUSDataset.objects.filter(name=name).count() >  0:
            raise forms.ValidationError(
                _('The name "%(name)" is already taken. Try picking a new one'),
                params={'name': name}
            )

        if not re.match(r'^\w[\w\d ]*$', name):
            raise forms.ValidationError(
                _('The name should only contains letters, numbers and spaces. '
                  'The first character must be letters.')
            )

        return name


class GetResultsForm(forms.Form):



    # Add a new field for private databases
    datasets = forms.ModelChoiceField(
        #queryset=BUSDataset.objects.filter(permission = 'private'),
        queryset=BUSDataset.objects.all(),
        label='Select Dataset',
        empty_label="Choose a dataset",
        required=True, # Ensure this is set to True
    )

   # Add a new field for private databases
    models = forms.ModelChoiceField(
        #queryset=BUSDataset.objects.filter(permission = 'private'),
        queryset=ModelCheckpoint.objects.all(),
        label='Select A Model To Perform Evaluation',
        empty_label="Choose a trained model",
        required=True, # Ensure this is set to True
    )


    def clean_name(self):
        name = self.cleaned_data.get('name')
        if len(name) <  4:
            raise forms.ValidationError(
                _('Dataset name must be at least  4 characters long'))

        if BUSDataset.objects.filter(name=name).count() >  0:
            raise forms.ValidationError(
                _('The name "%(name)" is already taken. Try picking a new one'),
                params={'name': name}
            )   

        if not re.match(r'^\w[\w\d ]*$', name):
            raise forms.ValidationError(
                _('The name should only contains letters, numbers and spaces. '
                  'The first character must be letters.')
            )

        return name

    def clean(self):
        name = self.cleaned_data.get('name')
        if name is None:
            return
        cleaned_data = super().clean()
        training_percentage = cleaned_data.get('training_percentage')
        validation_percentage = cleaned_data.get('validation_percentage')
        test_percentage = cleaned_data.get('test_percentage')

        total_percentage = training_percentage + validation_percentage + test_percentage
        if total_percentage !=  100:
            raise forms.ValidationError('The sum of training, validation, and test percentages must equal  100%.')

        self.cleaned_data['data_file'] = name
        return self.cleaned_data

