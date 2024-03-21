from django.db import models
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _
from django.dispatch import receiver
from django.db.models.signals import post_save
from .countries import COUNTRIES


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE,
                                verbose_name='User Profile',
                                primary_key=True,
                                related_name='profile')
    organization = models.CharField(_('Organization/Institution'), max_length=50,
                                    default='')
    country = models.CharField(_('Country'), max_length=5,
                               choices=COUNTRIES,
                               blank=True, null=True)

    def __str__(self):
        return 'Profile of user {}'.format(self.user.username)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        profile = UserProfile.objects.create(user=instance)
        profile.save()
