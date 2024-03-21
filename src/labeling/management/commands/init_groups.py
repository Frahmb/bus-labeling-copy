from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, User

GROUPS = ['labeling_pub_view', 'labeling_pub_edit', 'labeling_pub_download']


class Command(BaseCommand):
    help = "Initialize groups"

    def handle(self, *args, **options):
        for group_name in GROUPS:
            group, _ = Group.objects.get_or_create(name=group_name)
            for user in User.objects.all():
                user.groups.add(group)
