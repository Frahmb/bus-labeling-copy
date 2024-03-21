from django.contrib import admin
from django.http.request import HttpRequest
from django.urls import reverse
from django.utils.html import format_html
from .models import File, DownloadHistory


class DownloadHistoryInline(admin.TabularInline):
    model = DownloadHistory
    extra = 0
    readonly_fields = ('downloader', 'download_time')
    fields = ('downloader', 'download_time')
    ordering = ('-download_time',)

    def has_add_permission(self, request: HttpRequest, obj):
        return False
    
    def has_delete_permission(self, request: HttpRequest, obj):
        return False


@admin.register(File)
class FileAdmin(admin.ModelAdmin):
    list_display = ('filename', 'download_url', 'valid_until', 'download_num', 'uploader')
    list_filter = ('valid_until', 'uploader')
    search_fields = ('file', 'uploader__username')
    fields = ('id', 'filename', 'file', 'valid_until', 'uploader', 'download_num', 'download_url')
    readonly_fields = ('id', 'filename', 'uploader', 'download_num', 'download_url')
    ordering = ('-uploading_time',)
    inlines = (DownloadHistoryInline,)
    date_hierarchy = 'uploading_time'

    def download_url(self, obj):
        url = reverse('filesharing:download', args=[obj.id])
        return format_html('<a href="{}">Link</a>', url)

    # save uploader automatically
    def save_model(self, request, obj, form, change):
        if not change:
            obj.uploader = request.user
        super().save_model(request, obj, form, change)
