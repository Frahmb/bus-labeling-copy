import os
import json
import shutil
from functools import reduce
from django.db import models
from django.conf import settings
from django.contrib.auth.models import User, Group, Permission
from django.contrib.contenttypes.models import ContentType
from django.utils.html import format_html
from django.forms.models import model_to_dict
from django.apps import apps
from django.urls import reverse
from django.db import connection
from django.utils.translation import gettext_lazy as _, gettext
from django.contrib.auth.management import create_permissions
import pandas as pd


class BIRADSMarginChoice(models.Model):
    MARGIN_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('Circumscribed')),
        (_('Not circumscribed'), (
            (1, _('Indistinct')),
            (2, _('Angular')),
            (3, _('Microlobulated')),
            (4, _('Spiculated'))
        )),
    )
    choice = models.IntegerField(_('Margin'), choices=MARGIN_CHOICE, unique=True)

    def __str__(self):
        return '{}: {}'.format(self.choice, self.get_choice_display())

    @classmethod
    def init_db(cls):
        for i in range(-1, 5):
            cls(choice=i).save()


class BIRADS(models.Model):
    SHAPE_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('Oval')),
        (1, _('Round')),
        (2, _('Irregular')),
    )

    ORIENTATION_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('Parallel')),
        (1, _('Not Parallel'))
    )

    ECHO_PATTERN_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('Anechoic')),
        (1, _('Hyperechoic')),
        (2, _('Complex cystic and solid')),
        (3, _('Hypoechoic')),
        (4, _('Isoechoic')),
        (5, _('Heterogeneous')),
    )

    POSTERIOR_FEATURE_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('No posterior features')),
        (1, _('Enhancement')),
        (2, _('Shadowing')),
        (3, _('Combined pattern'))
    )

    CALCIFICATION_CHOICE = (
        (-1, _('UNKNOWN')),
        (0, _('Calcifications in a mass')),
        (1, _('Calcifications outside of a mass')),
        (2, _('Intraductal calcifications')),
        (3, _('NONE'))
    )

    BIRADS_ASSESSMENT_CHOICE = (
        ('-1', _('UNKNOWN')),
        ('0', _('Category 0: Incomplete - Need Additional Imaging')),
        ('1', _('Category 1: Negative')),
        ('2', _('Category 2: Benign')),
        ('3', _('Category 3: Probably Benign')),
        (_('Category 4: Suspicious'), (
            ('4A', _('Category 4A: Low suspicion for malignancy')),
            ('4B', _('Category 4B: Moderate suspicion for malignancy')),
            ('4C', _('Category 4C: High suspicion for malignancy')),
        )),
        ('5', _('Category 5: Highly suggestive of malignancy')),
        ('6', _('Category 6: Known biopsy-proven malignancy')),
    )

    id = models.AutoField(primary_key=True, verbose_name="ID")
    shape = models.IntegerField(_('Shape'),
                                choices=SHAPE_CHOICE,
                                null=True)
    orientation = models.IntegerField(_('Orientation'),
                                      choices=ORIENTATION_CHOICE,
                                      null=True)
    margin = models.ManyToManyField(BIRADSMarginChoice,
                                    verbose_name=_('Margin'))
    echo_pattern = models.IntegerField(_('Echo Pattern'),
                                       choices=ECHO_PATTERN_CHOICE,
                                       null=True)
    posterior_feature = models.IntegerField(
        _('Posterior Features'),
        choices=POSTERIOR_FEATURE_CHOICE,
        null=True)
    calcification = models.IntegerField(_('Calcification'),
                                        choices=CALCIFICATION_CHOICE,
                                        null=True)
    assessment = models.CharField(_('Assessment'), max_length=10,
                                  choices=BIRADS_ASSESSMENT_CHOICE)
    creator = models.ForeignKey(User, on_delete=models.CASCADE,
                                verbose_name=_('Creator'))
    completed = models.BooleanField(_('Completed'), default=False)

    diagnosis = models.BooleanField(_('Diagnosis'), default=True)   
    
    def export(self):
        record = model_to_dict(self)
        record['creator'] = self.creator.username
        record['margin'] = [i.choice for i in record['margin']]
        del record['id']
        del record['completed']
        return record

    def __str__(self):
        return 'id: {}, {}'.format(self.id, self.creator.username)


class Masking(models.Model):
    id = models.AutoField(primary_key=True, verbose_name="ID")
    cropping = models.TextField(_('Cropping'), default='')
    tumor = models.TextField(_('Tumor Mask'), default='')
    tissue = models.TextField(_('Tissue Mask'), default='')
    creator = models.ForeignKey(User, on_delete=models.CASCADE)

    def export(self):
        record = model_to_dict(self)
        record['creator'] = self.creator.username
        for i in ['cropping', 'tumor', 'tissue']:
            record[i] = json.dumps(record[i]) if record[i] else None
        del record['id']
        return record

    def __str__(self):
        return 'id: {}, {}'.format(self.id, self.creator.username)


class BaseCase(models.Model):
    id = models.AutoField(primary_key=True, verbose_name="ID")
    birads = models.ManyToManyField(BIRADS,
                                    verbose_name=_('BIRADS'))  
     
    # Tumor Type
    TYPE_CHOICE = (
        ('N', 'Normal'),
        ('B', 'Benign'),
        ('M', 'Malignant'),
        ('O', 'Others'),
        ('U', 'UNKNOWN'),
    )
    tumor_type = models.CharField('Tumor Type', max_length=10,
                                  choices=TYPE_CHOICE)

    def image_preview(self):
        images = self.images.filter(applicable=True)
        if images.count() > 0:
            images = images.all()
        else:
            images = self.images.all()
        if images.count() == 0:
            return
        return reduce(
            lambda x, y: x + format_html('&nbsp') + y, [i.image_preview()
                                                        for i in images])

    image_preview.short_description = _('Image/Annotation Preview')

    def image_count(self):
        return self.images.count()

    image_count.short_description = _('Image Count')

    def applicable_image_count(self):
        return self.images.filter(applicable=True).count()

    applicable_image_count.short_description = _('Applicable Image Count')

    def export(self, user=None):
        record = model_to_dict(self)

        if user is None:
            birads = self.birads.filter(completed=True).all()
        else:
            birads = self.birads.filter(creator=user, completed=True).all()

        record['birads'] = [i.export() for i in birads]
        record['images'] = [i.export(user) for i in self.images.all()]
        return record

    def __str__(self):
        return str(self.id)

    @classmethod
    def get_extra_fields(cls):
        extra_cols = [f.attname for f in cls._meta.fields if f.attname.startswith('extra')]
        extra_cols.insert(0, 'tumor_type')
        return extra_cols

    class Meta:
        abstract = True


class BaseImage(models.Model):
    id = models.AutoField(primary_key=True, verbose_name="ID")
    image = models.ImageField(_('Image'))
    masking = models.ManyToManyField(Masking, verbose_name=_('Masking'))
    applicable = models.BooleanField(_('Applicable'), default=True)
    case = models.ForeignKey(BaseCase, models.CASCADE, related_name='images',
                             verbose_name=_('Case'))

    def image_preview(self):
        img_tag = format_html(f'<img src="{self.image.url}"'
                              'style="max-height: 400px; max-width: 400px;" />')
        if hasattr(self, 'mask'):
            mask_tag = format_html(f'<img src="{self.mask.url}"'
                                   'style="max-height: 400px; max-width: 400px;" />')
            return img_tag + format_html('&nbsp') + mask_tag
        else:
            return img_tag

    image_preview.short_description = _('Image/Annotation Preview')

    def description(self):
        return 'ID: {}, Img: {}'.format(
            self.id, self.image.name
        )

    description.short_description = _('Description')

    def get_mask(self, user):
        query = self.masking.filter(creator=user)
        if query.count() == 0:
            masking = Masking(creator=user)
            masking.save()
            self.masking.add(masking)
            self.save()
            return masking
        else:
            return query.first()

    @classmethod
    def get_next_unlabeled(cls, user, mask_type, current_id=-1):
        query = cls.objects.filter(id__gt=current_id, applicable=True)
        created_objects = query.filter(**{
            f'masking__{mask_type}': '',
            'masking__creator': user
        })
        uncreated_objects = query.exclude(**{
            'masking__creator': user
        })
        all_objects = created_objects.union(
            uncreated_objects).distinct().order_by('id')
        return all_objects.first() if all_objects.count() > 0 else None

    @classmethod
    def get_next(cls, obj):
        return cls.objects.filter(id__gt=obj.id, applicable=True).first()

    @classmethod
    def get_prev(cls, obj):
        return cls.objects.filter(id__lt=obj.id, applicable=True).last()

    def mask_cropping(self):
        if self.id is None:
            return
        url = reverse(
            'masking_change',
            args=('cropping', self._meta.model_name, self.id)
        )
        return format_html(
            f'<a href="{url}">' + gettext('Cropping') + '</a>')

    mask_cropping.short_description = _('Cropping')

    def mask_tumor(self):
        if self.id is None:
            return
        url = reverse(
            'masking_change',
            args=('tumor', self._meta.model_name, self.id)
        )
        return format_html(
            f'<a href="{url}">' + gettext('Mask Tumor') + '</a>')

    mask_tumor.short_description = _('Mask Tumor')

    def mask_tissue(self):
        if self.id is None:
            return
        url = reverse(
            'masking_change',
            args=('tissue', self._meta.model_name, self.id)
        )
        return format_html(
            f'<a href="{url}">' + gettext('Mask Tissue') + '</a>')

    mask_tissue.short_description = _('Mask Tissue')

    def export(self, user=None):
        record = model_to_dict(self)
        record['image'] = self.image.name
        if hasattr(self, 'mask'):
            record['mask'] = self.mask.name

        if user is None:
            masking = self.masking.all()
        else:
            masking = self.masking.filter(creator=user)

        record['masking'] = [i.export() for i in masking]

        del record['id']
        return record

    def __str__(self):
        return self.image.name

    class Meta:
        abstract = True


class BUSDataset(models.Model):
    GROUPS = ['labeling_pub_view', 'labeling_pub_edit', 'labeling_pub_download']

    id = models.AutoField(primary_key=True, verbose_name="ID")
    name = models.CharField(verbose_name=_('Dataset Name'), max_length=120, unique=True)
    creator = models.ForeignKey(User, on_delete=models.CASCADE,
                                verbose_name=_('Creator'))
    path = models.FilePathField(verbose_name=_('Dataset Path'),
                                path=settings.MEDIA_ROOT,
                                allow_files=False, allow_folders=True)

    PERM = (
        ('no', 'Private (only the uploader can see)'),
        ('view', 'Everyone can view'),
        ('edit', 'Everyone can edit/label'),
        ('download', 'Everyone can download'),
    )
    permission = models.CharField(verbose_name=_('Permission'),
                                  choices=PERM, max_length=20,
                                  default='no')

    model_cache = {}

    def delete(self, using=None, keep_parents=False):
        # unregister from admin
        self.unregister_admin_model()

        case_model, image_model = self.get_models()

        # remove models from model_cache
        del self.model_cache[self.safe_name + 'Case']
        del self.model_cache[self.safe_name + 'Image']

        # delete permission entries
        ContentType.objects.get(model=case_model._meta.model_name).delete()
        ContentType.objects.get(model=image_model._meta.model_name).delete()

        # delete case and image tables
        with connection.schema_editor() as schema_editor:
            schema_editor.delete_model(image_model)
            schema_editor.delete_model(case_model)
        
        # delete models
        del case_model, image_model

        # delete folder
        shutil.rmtree(os.path.join(settings.MEDIA_ROOT, self.path))

        # delete entry itself
        super().delete(using, keep_parents)

    @property
    def safe_name(self):
        """return a safe class name"""
        safe_name = ''.join([c for c in self.name if c.isalpha() or c.isdigit()])
        if not safe_name[0].isalpha():
            # add a prefix when the first letter is not alpha
            safe_name = 'bus' + safe_name
        safe_name += self.path.split('_')[-1][:4]
        return safe_name

    def parse_scheme(self):
        df = pd.read_csv(os.path.join(
            settings.MEDIA_ROOT,
            self.path,
            'label.csv'
        ))
        scheme = {}

        for i in range(3, len(df.columns)):
            field_name = f'extra_col_{i}'
            col_name = df.columns[i]
            col = df.iloc[:, i]

            if pd.api.types.is_integer_dtype(col.dtype):
                scheme[field_name] = models.IntegerField(verbose_name=col_name)
            elif pd.api.types.is_float_dtype(col.dtype):
                scheme[field_name] = models.FloatField(verbose_name=col_name)
            else:
                scheme[field_name] = models.TextField(verbose_name=col_name)
        return scheme

    def create_case_model(self):
        class_name = self.safe_name + 'Case'
        if class_name in self.model_cache:
            return self.model_cache[class_name]

        class Meta:
            app_label = 'labeling'
            verbose_name = self.name
            verbose_name_plural = self.name
            permissions = (
                ("download_" + class_name.lower(), "Can download" + class_name),
            )

        attrs = {'__module__': __name__, 'Meta': Meta, 'dataset_id': self.id}
        attrs.update(self.parse_scheme())
        case_model = type(class_name, (BaseCase,), attrs)
        self.model_cache[class_name] = case_model
        setattr(case_model, '_ds_model', self)
        return case_model

    def create_image_model(self, case_model):
        class_name = self.safe_name + 'Image'
        if class_name in self.model_cache:
            return self.model_cache[class_name]

        class Meta:
            app_label = 'labeling'
            verbose_name = self.name + ' Image'
            verbose_name_plural = self.name + ' Images'
            permissions = (
                ("download_" + class_name.lower(), "Can download " + class_name),
            )

        attrs = {
            '__module__': __name__,
            'Meta': Meta,
            'case': models.ForeignKey(case_model, models.CASCADE, related_name='images')}
        image_model = type(class_name, (BaseImage,), attrs)
        self.model_cache[class_name] = image_model
        setattr(case_model, '_image_model', image_model)
        setattr(image_model, '_case_model', case_model)
        setattr(image_model, '_ds_model', self)
        return image_model

    def get_models(self):
        case_model = self.create_case_model()
        image_model = self.create_image_model(case_model)
        return case_model, image_model

    def register_admin_model(self):
        case_model, image_model = self.get_models()

        from django.contrib import admin
        from .admin import BaseCaseAdmin, BaseImageAdmin, labeling_site

        @admin.register(case_model, site=labeling_site)
        class CaseAdmin(BaseCaseAdmin):
            pass

        @admin.register(image_model, site=labeling_site)
        class ImageAdmin(BaseImageAdmin):
            pass

    def unregister_admin_model(self):
        from .admin import labeling_site
        labeling_site.unregister(self.get_models())

    def load_dataset(self):
        case_model, image_model = self.get_models()

        df = pd.read_csv(os.path.join(
            settings.MEDIA_ROOT,
            self.path,
            'label.csv'
        ))
        grouped = df.groupby('case_id')
        for case_id, group in grouped:
            case = case_model(id=case_id,
                              tumor_type=group.iloc[0, 2])

            # set value for extra columns: extra_col1, extra_col2, ...
            for name, field in self.parse_scheme().items():
                value = group[field.verbose_name].iloc[0]
                if isinstance(field, models.IntegerField):
                    value = int(value)
                elif isinstance(field, models.FloatField):
                    value = float(value)
                else:
                    value = str(value)
                setattr(case, name, value)

            case.save()
            for _, row in group.iterrows():
                img = image_model(image=os.path.join(self.path, row['image']),
                                  case=case)
                img.save()

    def create_db_table(self):
        # create tables
        for model in self.get_models():
            with connection.schema_editor() as schema_editor:
                schema_editor.create_model(model)

        # create permissions
        create_permissions(apps.get_app_config('labeling'),
                           interactive=False)

        # grant the creator all permissions
        self.creator.user_permissions.add(*self.get_permissions('case'))
        self.creator.user_permissions.add(*self.get_permissions('image'))

        # set permissions for other groups
        self.set_permission()

    def get_permissions(self, model='case', permission='all'):
        case_model, image_model = self.get_models()
        model_name = case_model._meta.model_name if model == 'case' else \
            image_model._meta.model_name
        ct = ContentType.objects.get(model=model_name)
        perms = Permission.objects.filter(content_type=ct)

        if permission == 'all':
            return perms.all()
        else:
            return perms.get(codename=f'{permission}_{model_name}')

    def set_permission(self):
        # revoke all permissions from all groups
        for group_name in self.GROUPS:
            group = Group.objects.get(name=group_name)
            group.permissions.remove(*self.get_permissions('case'))
            group.permissions.remove(*self.get_permissions('image'))

        # grant permissions
        if self.permission == 'view':
            group = Group.objects.get(name='labeling_pub_view')
            group.permissions.add(self.get_permissions('case', 'view'))
            group.permissions.add(self.get_permissions('image', 'view'))
        elif self.permission == 'edit':
            group = Group.objects.get(name='labeling_pub_edit')
            group.permissions.add(self.get_permissions('case', 'view'))
            group.permissions.add(self.get_permissions('case', 'change'))
            group.permissions.add(self.get_permissions('image', 'view'))
            group.permissions.add(self.get_permissions('image', 'change'))
        elif self.permission == 'download':
            group = Group.objects.get(name='labeling_pub_download')
            group.permissions.add(self.get_permissions('case', 'view'))
            group.permissions.add(self.get_permissions('case', 'change'))
            group.permissions.add(self.get_permissions('case', 'download'))
            group.permissions.add(self.get_permissions('image', 'view'))
            group.permissions.add(self.get_permissions('image', 'change'))
            group.permissions.add(self.get_permissions('image', 'download'))

    def save(self, force_insert=False, force_update=False, using=None,
             update_fields=None):
        if self.id is not None:
            self.set_permission()
        return super().save(force_insert, force_update, using, update_fields)

    def __str__(self):
        return self.name
