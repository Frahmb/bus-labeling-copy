import os
from django.contrib.auth.models import User, Group
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.utils.translation import gettext_lazy as _


class UploadingFile(models.Model):
    id = models.AutoField(primary_key=True)
    file = models.FileField(verbose_name='File',
                            upload_to='uploading/%Y/%m/%d/',
                            blank=True)
    uploading_time = models.DateTimeField(verbose_name='Uploading Time',
                                          auto_now_add=True,
                                          blank=True)
    uploader = models.ForeignKey(User, related_name='uploaded_files',
                                 on_delete=models.CASCADE, editable=False)
    allowed_users = models.ManyToManyField(User, related_name='allowed_files',
                                           blank=True)
    allowed_groups = models.ManyToManyField(Group, blank=True)

    def __str__(self):
        return os.path.basename(self.file.name)

    class Meta:
        verbose_name = "File"
        verbose_name_plural = "Files"


@receiver(post_delete, sender=UploadingFile)
def delete_file(sender, instance, using, **kwargs):
    if instance.file and os.path.isfile(instance.file.path):
        os.remove(instance.file.path)


class Competition(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(verbose_name=_('Name'), max_length=100)
    description = models.TextField(verbose_name=_('Description'))
    cover_img = models.FileField(verbose_name=_('Cover Image'),
                                 upload_to='uploading/competition/',
                                 blank=True)
    end_date = models.DateTimeField(verbose_name=_("End Date"),
                                    blank=True, null=True)
    submit_file = models.BooleanField(verbose_name=_('Submit File'),
                                      default=False)
    # 'evaluation_func' stores a function that can be used to evaluate the submission
    evaluation_func = models.TextField(verbose_name=_("Evaluation Function"),
                                       blank=True)
    # 'labels' stores the GT of the competition, separated by comma
    labels = models.TextField(verbose_name=_("Labels"),
                              blank=True)
    download_link = models.CharField(verbose_name=_('Download Link'),
                                     max_length=150,
                                     blank=True)

    def __str__(self):
        return self.name


class CompetitionSubmission(models.Model):
    id = models.AutoField(primary_key=True)
    competition = models.ForeignKey(Competition,
                                    on_delete=models.CASCADE,
                                    verbose_name=_('Competition'))
    creator = models.ForeignKey(User, on_delete=models.CASCADE,
                                verbose_name=_('Submitter'))
    content = models.TextField(verbose_name=_('Content'),
                               blank=True, null=True, default=None)
    submit_file = models.FileField(verbose_name=_('Submit File'),
                                   upload_to='competition/submissions/%Y/%m/%d/',
                                   blank=True,
                                   null=True,
                                   default=None)
    score = models.CharField(max_length=100, verbose_name=_('Score'))
    submit_time = models.DateTimeField(verbose_name=_('Submit Time'),
                                       auto_now_add=True)

    def __str__(self):
        return str(self.id)
