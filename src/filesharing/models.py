import os, uuid
from django.db import models
from django.urls import reverse
from django.contrib.auth.models import User


class File(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    file = models.FileField(verbose_name='File',
                            upload_to='uploading/%Y/%m/%d/')
    uploading_time = models.DateTimeField(verbose_name='Uploading Time',
                                          auto_now_add=True)
    uploader = models.ForeignKey(User,
                                 on_delete=models.CASCADE,
                                 editable=False)
    valid_until = models.DateTimeField(verbose_name='Valid Until',
                                       null=True,
                                       blank=True)

    class Meta:
        verbose_name = "File"
        verbose_name_plural = "Files"

    def __str__(self):
        return os.path.basename(self.file.name)
    
    @property
    def filename(self):
        return os.path.basename(self.file.name)
    
    @property
    def download_num(self):
        return DownloadHistory.objects.filter(file=self).count()

    def track_download(self, downloader: User):
        DownloadHistory.objects.create(file=self, downloader=downloader)


class DownloadHistory(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE)
    downloader = models.ForeignKey(User, on_delete=models.CASCADE)
    download_time = models.DateTimeField(auto_now_add=True)
