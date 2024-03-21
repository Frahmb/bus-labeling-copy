from django.contrib import admin
from django.http import HttpResponseRedirect
from django.db import models
from django.utils.html import format_html
from django.urls import reverse
from django.contrib.admin.widgets import FilteredSelectMultiple
from .models import UploadingFile, Competition, CompetitionSubmission


@admin.register(UploadingFile)
class UploadingFile(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploading_time', 'uploader')
    readonly_fields = ('id', 'uploading_time', 'uploader')
    save_as = False
    formfield_overrides = {
        models.ManyToManyField: {
            'widget': FilteredSelectMultiple('options', is_stacked=False)},
    }

    def has_add_permission(self, request):
        return True

    def has_change_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        elif obj is None:
            return False
        elif obj.uploader == request.user:
            return True

    def has_view_permission(self, request, obj=None):
        return True

    def has_delete_permission(self, request, obj=None):
        if request.user.is_superuser:
            return True
        elif obj is None:
            return False
        elif obj.uploader == request.user:
            return True

    def get_queryset(self, request):
        # For non-root user, only return related files
        if request.user.is_superuser:
            qs = super().get_queryset(request)
            return qs
        else:
            qs1 = request.user.allowed_files.all()
            qs2 = request.user.uploaded_files.all()
            qs = (qs1 | qs2).distinct()
            return qs

    def save_model(self, request, obj, form, change):
        obj.uploader = request.user
        obj.save()

    def response_add(self, request, obj, post_url_continue=None):
        res = super().response_add(request, obj, post_url_continue)
        return HttpResponseRedirect('/labeling')

    def get_readonly_fields(self, request, obj=None):
        if obj:
            return super().get_readonly_fields(request, obj)
        else:
            return tuple()


@admin.register(CompetitionSubmission)
class CompetitionSubmissionAdmin(admin.ModelAdmin):
    list_display = ('id', 'creator', 'score', 'competition', 'submit_time')


class CompetitionSubmissionInline(admin.TabularInline):
    model = CompetitionSubmission
    fields = ('id', 'creator', 'submit_time', 'score')
    readonly_fields = fields
    ordering = ['-score']
    show_change_link = True

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def has_add_permission(self, request, obj):
        return False


@admin.register(Competition)
class CompetitionAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'end_date')
    inlines = [CompetitionSubmissionInline]
    fields = (
        'id', 'url', 'name', 'description', 'cover_img',
        'submit_file', 'end_date',
        'evaluation_func', 'labels', 'download_link')
    
    readonly_fields = ('id', 'url')

    def url(self, obj):
        if obj.id is None:
            return ''
        
        url = reverse('competition', args=[obj.id])
        return format_html(
            '<a href="%s">Link</a>' % (url,)
        )
