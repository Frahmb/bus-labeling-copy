from django.shortcuts import render
from importlib import reload, import_module
from collections import OrderedDict
from typing import Any, Type
from django.conf import settings
from django.contrib import admin
from django.db.models.query import QuerySet
from django.http.request import HttpRequest
from django.utils.html import format_html
from django.contrib.auth.models import User
from django.urls import reverse, clear_url_caches, NoReverseMatch
from django.contrib.admin.models import LogEntry
from django.forms import CheckboxSelectMultiple
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.db import models
from django.apps import apps
from django.utils.text import capfirst
from django.utils.translation import gettext_lazy as _
from user.models import UserProfile
from .forms import UploadingDatasetForm
from .forms import SplittingDatasetForm
from .forms import ModelRetrainForm
from .actions import export_self_labeled, export_all_labeled, split_tr_v_t
from .models import (
    BaseCase, BaseImage,
    BIRADS, Masking,
    BUSDataset,
    SplitDataset
)


class AdminSite(admin.AdminSite):
    site_header = _('MIDA Labeling System')
    site_title = _('MIDA Labeling System')
    index_title = _('Labeling System')

    def get_app_list(self, request):
        """
        Return a sorted list of all the installed apps that have been
        registered in this site.
        """
        app_dict = self._build_app_dict(request)

        # Sort the apps alphabetically.
        app_list = app_dict.values()

        # Sort the models alphabetically within each app.
        for app in app_list:
            app['models'].sort(key=lambda x: x['name'])

        return app_list

    def _build_app_dict(self, request, label=None):
        """
        Build the app dictionary. The optional `label` parameter filters models
        of a specific app.
        """

        if label is None:
            return OrderedDict(
                [(label, self._build_app_dict(request, label))
                 for label in ['labeling', 'public', 'private']]
            )

        app_dict = {
            'name': apps.get_app_config(label).verbose_name,
            'app_label': label,
            'app_url': '',
            'has_module_perms': True,
            'models': [],
        }

        models = self._registry

        for model, model_admin in models.items():
            app_label = model._meta.app_label

            if not hasattr(model, 'dataset_id'):
                continue

            try:
                ds = BUSDataset.objects.get(id=model.dataset_id)
            except:
                continue

            if label == 'labeling':
                if ds.creator != request.user:
                    continue

            elif label == 'public':
                if ds.permission == 'no' or ds.creator == request.user:
                    continue

            elif label == 'private':
                if ds.permission == 'no' and ds.creator != request.user:
                    model_dict = {
                        'name': capfirst(model._meta.verbose_name_plural),
                        'object_name': model._meta.object_name,
                        'admin_url': None,
                        'add_url': None,
                        'view_only': True,
                        'owner_url': None,
                        'mask_url': None,
                        'creator': ds.creator.username
                    }
                    app_dict['models'].append(model_dict)
                continue

            perms = model_admin.get_model_perms(request)
            if True not in perms.values():
                continue

            info = (app_label, model._meta.model_name)
            model_dict = {
                'name': capfirst(model._meta.verbose_name_plural),
                'object_name': model._meta.object_name,
                'perms': perms,
                'admin_url': None,
                'add_url': None,
                'view_only': True,
                'owner_url': None,
                'mask_url': None,
                'creator': ds.creator.username
            }

            try:
                model_dict['admin_url'] = reverse('admin:%s_%s_changelist' % info, current_app=self.name)
            except NoReverseMatch:
                pass

            if model._ds_model.creator == request.user:
                model_dict['owner_url'] = reverse(
                    'labeling:{}_{}_change'.format(app_label, model._ds_model._meta.model_name),
                    args=(model._ds_model.id,)
                )

            if request.user.has_perm(
                    'labeling.change_{}'.format(model._meta.model_name)):
                model_dict['mask_url'] = reverse('masking', args=('cropping', model._image_model._meta.model_name))

            app_dict['models'].append(model_dict)

        return app_dict





    def get_urls(self):
        from django.urls import path

        urls = super().get_urls()
        urls.insert(
            0,
            path(
                'uploading/',
                 self.admin_view(self.uploading), name='uploading'
            )
        )
        urls.insert(
            1,  # Insert the splitting view just after the uploading view
            path(
                'splitting/',
                self.admin_view(self.splitting), name='splitting'
            )
        )
        urls.insert(
            2,
            path(
                'retrain/',
                self.admin_view(self.retrain), name='retrain'
            )
        )
        return urls 
    
    def splitting(self, request):
       # app_dict = self._build_app_dict(request, label='labeling')
        
        if request.method == 'POST':
            form = SplittingDatasetForm(request.POST, request.FILES)
            context = {
                **self.each_context(request),
                'title': 'Split Dataset',
                'form': form
                }
            if form.is_valid():
                print("got here")
                #selecting parts of the form we need
                name = form.cleaned_data['name']
                dataset_id = form.cleaned_data['private_database']
                training_percentage = form.cleaned_data['training_percentage']
                validation_percentage = form.cleaned_data['validation_percentage']
                test_percentage = form.cleaned_data['test_percentage']

                # Retrieve the dataset
                dataset = BUSDataset.objects.get(name=dataset_id)
                dataset.load_dataset()
                print("got here!'")
                # split the dataset using the function defined in actions.py
                df_input = dataset.to_dataframe()

                df_train, df_val, df_test = split_tr_v_t(df_input, training_percentage/100, validation_percentage/100, test_percentage/100)

                # Serialize the dataframes to CSV format
                training_data = df_train.to_csv(index=False)
                validation_data = df_val.to_csv(index=False)
                test_data = df_test.to_csv(index=False)

                print("Got here???")

              # Create new SplitDataset instances for each split
                training_split = SplitDataset.objects.create(
                    name=f'{name}_training',
                    original_dataset=dataset,
                    percentage=training_percentage,
                    data=training_data
                )
                validation_split = SplitDataset.objects.create(
                    name=f'{name}_validation',
                    original_dataset=dataset,
                    percentage=validation_percentage,
                    data=validation_data
                )  
                test_split = SplitDataset.objects.create(
                    name=f'{name}_test',
                    original_dataset=dataset,
                    percentage=test_percentage,
                    data=test_data
                )

                # Save the split datasets to the database
                training_split.save()
                validation_split.save()
                test_split.save()

                # Redirect or render a success message
                print("we got here")
                return HttpResponseRedirect(reverse('labeling:index'))
            else:
                print("form is invalid") 
            # Split the dataset into training, validation, and test datasets
        else:
            print("no dice")
            form = SplittingDatasetForm()

        # If the form is not valid, re-render the form with the errors
        context = {
            **self.each_context(request),
            'title': 'Split Dataset',
            'form': form
        }
        return render(request, 'admin/split_dataset.html', context)
                
                



    def uploading(self, request):
        if request.method == 'POST':
            form = UploadingDatasetForm(request.POST, request.FILES)
            if form.is_valid():
                name = form.cleaned_data['name']
                data_file = form.cleaned_data['data_file']
                permission = form.cleaned_data['public_permission']

                ds = BUSDataset(name=name,
                                creator=request.user,
                                permission=permission,
                                path=data_file)
                ds.save()
                ds.create_db_table()
                ds.register_admin_model()
                ds.load_dataset()

                reload(import_module(settings.ROOT_URLCONF))
                clear_url_caches()
                return HttpResponseRedirect(reverse('labeling:index'))
        else:
            form = UploadingDatasetForm()

        from django.shortcuts import render

        context = {
            **self.each_context(request),
            'title': 'Upload Dataset',
            'form': form
        }

        return render(request, 'admin/upload_dataset.html', context)
    

    def retrain(self, request):
        if request.method == 'POST':
            form = UploadingDatasetForm(request.POST, request.FILES)
            if form.is_valid():
                name = form.cleaned_data['name']
                data_file = form.cleaned_data['data_file']
                permission = form.cleaned_data['public_permission']

                ds = BUSDataset(name=name,
                                creator=request.user,
                                permission=permission,
                                path=data_file)
                ds.save()
                ds.create_db_table()
                ds.register_admin_model()
                ds.load_dataset()

                reload(import_module(settings.ROOT_URLCONF))
                clear_url_caches()
                return HttpResponseRedirect(reverse('labeling:index'))
        else:
            form = ModelRetrainForm()

        from django.shortcuts import render

        context = {
            **self.each_context(request),
            'title': 'Retrain Models',
            'form': form
        }

        return render(request, 'admin/retrain_model.html', context)
    



    def index(self, request, extra_context=None):
        if extra_context is None:
            extra_context = {}

        extra_context['total_users'] = User.objects.count()
        extra_context['recent_users'] = User.objects.order_by('-last_login')[:5]
        return super().index(request, extra_context)


labeling_site = AdminSite(name='labeling')
admin.site.site_header = 'MIDA Labeling System'
admin.site.site_title = 'MIDA Labeling System'
admin.site.index_title = 'Labeling System'


@admin.register(BIRADS)
@admin.register(BIRADS, site=labeling_site)
class BIRADSAdmin(admin.ModelAdmin):
    list_display = ('id', 'shape', 'orientation', 'echo_pattern',
                    'calcification', 'posterior_feature',
                    'assessment', 'creator')
    exclude = ('creator', 'completed')
    formfield_overrides = {
        models.ManyToManyField: {'widget': CheckboxSelectMultiple}
    }

    def save_model(self, request, obj, form, change):
        obj.creator = request.user
        obj.completed = True
        obj.save()

    def has_module_permission(self, request):
        return False


@admin.register(Masking)
@admin.register(Masking, site=labeling_site)
class MaskingAdmin(admin.ModelAdmin):
    readonly_fields = ('creator', )

    def has_module_permission(self, request):
        return False

    def save_model(self, request, obj, form, change):
        obj.creator = request.user
        obj.save()


def birads_inline(case_model: Type[BaseCase]):
    class BIRADSInline(admin.TabularInline):
        model = case_model.birads.through
        extra = 0

        def has_add_permission(self, request, obj):
            return False

    return BIRADSInline


def image_inline(image_model: Type[BaseImage]):
    class ImageInline(admin.StackedInline):
        model = image_model
        extra = 0
        exclude = ('masking',)
        readonly_fields = ('image', 'image_preview',
                           'mask_cropping', 'mask_tumor', 'mask_tissue')
        show_change_link = True
        fields = ('mask_cropping', 'mask_tumor', 'mask_tissue',
                  'image_preview', 'applicable')

        def has_add_permission(self, request, obj):
            return False

        def has_delete_permission(self, request, obj=None):
            return False
    return ImageInline


class BaseCaseAdmin(admin.ModelAdmin):
    exclude = ('birads',)
    search_fields = ('id',)
    ordering = ('id',)
    other_fields = tuple()

    list_display = ('id', 'image_count', 'applicable_image_count',
                    'birads_label_status', 'remaining_cropping',
                    'remaining_tumor_mask', 'remaining_tissue_mask', 'diagnosis')
    list_per_page = 25
    actions = [export_self_labeled, export_all_labeled]
    change_form_template = 'admin/case_change_form.html'
    change_list_template = 'admin/case_change_list.html'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.other_fields = self.model.get_extra_fields()

    @property
    def fieldsets(self):
        return (
            (None, {
                'fields': ('id', 'image_preview', 'birads_label')
            }),
            (_('Other Information'), {
                'fields': self.other_fields,
                'classes': ('collapse',),
            })
        )

    def change_view(self, request, object_id, form_url='', extra_context=None):
        image_model = self.model.images.rel.related_model
        self.inlines = [image_inline(image_model)]
        extra_context = extra_context or {}
        extra_context['title'] = self.model._meta.verbose_name
        ret = super().change_view(request, object_id, form_url, extra_context)
        return ret

    def get_readonly_fields(self, request, obj=None):
        ret = super().get_readonly_fields(request, obj)
        return list(ret) + ['id', 'image_preview', 'birads_label'] + \
               list(self.other_fields)

    def get_fieldsets(self, request, obj=None):
        ret = super().get_fieldsets(request, obj)
        if obj is None:
            setattr(self, 'birads', None)
            return ret

        # find the related birads instance
        # if it's not exist, create one
        birads = obj.birads.filter(creator=request.user).first()
        if birads is None:
            birads = BIRADS(creator=request.user)
            birads.save()
            obj.birads.add(birads)
            obj.save()

        setattr(self, 'birads', birads)
        return ret

    def birads_label(self, obj):
        if getattr(self, 'birads', None) is None:
            return None

        url = reverse('labeling:labeling_birads_change',
                      args=(self.birads.id,))

        return format_html(
            f'<a id="birads_label" href="{url}?_popup=1">BI-RADS</a>')

    birads_label.short_description = _('BI-RADS Label')

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changeform_view(self, request, object_id=None, form_url='', extra_context=None):
        extra_context = extra_context or {}
        extra_context['is_case'] = True
        return super().changeform_view(request, object_id, form_url, extra_context)

    def response_change(self, request, obj):
        res = super().response_change(request, obj)
        if '_save_and_next' in request.POST:
            next_obj = self.model.objects.filter(id__gt=obj.id).first()
            if next_obj:
                meta = obj._meta
                return HttpResponseRedirect(reverse(
                    'labeling:{}_{}_change'.format(
                        meta.app_label, meta.model_name),
                    args=(next_obj.id,)))
        return res

    def get_changelist(self, request, **kwargs):
        self.user = request.user
        return super().get_changelist(request, **kwargs)

    def birads_label_status(self, obj):
        birads = obj.birads.filter(creator=self.user).first()
        if birads is None:
            return False
        else:
            return birads.completed

    birads_label_status.short_description = _('BIRADS Label Status')
    birads_label_status.boolean = True

    def get_remaining(self, obj, name):
        query = obj.images.filter(applicable=True)
        remaining = 0
        for img in query:
            masking = img.masking.filter(creator=self.user).first()
            if masking is None or not getattr(masking, name):
                remaining += 1

        total = query.count()
        if total == 0:
            style = 'gary'
        elif remaining == 0:
            style = 'green'
        elif remaining == total:
            style = 'red'
        else:
            style = 'orange'

        return format_html(
            f'<span style="color: {style}">{remaining}/{query.count()}</span>')

    def remaining_cropping(self, obj):
        return self.get_remaining(obj, 'cropping')

    remaining_cropping.short_description = _('Remaining Cropping')

    def remaining_tumor_mask(self, obj):
        return self.get_remaining(obj, 'tumor')

    remaining_tumor_mask.short_description = _('Remaining Tumor Mask')

    def remaining_tissue_mask(self, obj):
        return self.get_remaining(obj, 'tissue')

    remaining_tissue_mask.short_description = _('Remaining Tissue Mask')

    def diagnosis(self, obj):
       
        # Get the type choice value for the current object
        type_choice = dict(BaseCase.TYPE_CHOICE).get(obj.tumor_type, 'UNKNOWN')
        return type_choice
    diagnosis.short_description = _('Diagnosis')


    def changelist_view(self, request, extra_context=None):
        ctx = {
            'image_model_name': self.model._image_model._meta.model_name.lower(),
            'has_change_permission': False
        }
        if request.user.has_perm(
            'labeling.change_{}'.format(self.model._meta.model_name)
        ):
            ctx['has_change_permission'] = True

        if extra_context is None:
            extra_context = ctx
        else:
            extra_context.update(ctx)

        extra_context['title'] = self.model._meta.verbose_name
        return super().changelist_view(request, extra_context)

    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.has_perm('labeling.download_{}'.format(self.model._meta.model_name)):
            return []
        else:
            return actions


class BaseImageAdmin(admin.ModelAdmin):
    search_fields = ('id',)
    exclude = ('masking',)
    fields = ('id', 'case_link', 'image_preview',
              'mask_cropping', 'mask_tumor', 'mask_tissue', 'applicable')
    readonly_fields = ('id', 'case', 'image_preview', 'case_link',
                       'mask_cropping', 'mask_tumor', 'mask_tissue')
    list_display = ('id',)
    ordering = ('id',)

    @staticmethod
    def case_link(obj):
        meta = obj.case._meta
        url = reverse(
            'labeling:{}_{}_change'.format(
                meta.app_label, meta.model_name),
            args=(obj.case,))
        return format_html(f'<a href="{url}">{obj.case}</a>')

    case_link.short_description = _('Case Link')

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_module_permission(self, request):
        return False


@admin.register(BUSDataset)
@admin.register(BUSDataset, site=labeling_site)
class BUSDatasetAdmin(admin.ModelAdmin):
    readonly_fields = ('name', 'path', 'creator')
    fields = ('name', 'path', 'creator', 'permission')
    list_display = ('name', 'creator', 'permission')

    def has_module_permission(self, request):
        return request.user.is_superuser

    def has_delete_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        else:
            return obj is not None and obj.creator == request.user

    def has_view_permission(self, request, obj=None):
        return self.has_delete_permission(request, obj)

    def has_change_permission(self, request, obj=None):
        return self.has_delete_permission(request, obj)

    def has_add_permission(self, request):
        return False

    def response_change(self, request, obj):
        return redirect('/')

    def response_delete(self, request, obj_display, obj_id):
        return redirect('/')
    
    def delete_queryset(self,
                        request: HttpRequest,
                        queryset: QuerySet[Any]) -> None:
        for obj in queryset:
            obj.delete()


@admin.register(LogEntry)
class LogAdmin(admin.ModelAdmin):
    def has_add_permission(self, request):
        return request.user.is_superuser

    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
