import os
import zipfile
import json
from io import BytesIO
from django.http import HttpResponse
from django.conf import settings
from django.utils.translation import gettext_lazy as _


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


export_self_labeled.short_description = _('Export your labels')


def export_all_labeled(modeladmin, request, queryset):
    return export_case(queryset)


export_all_labeled.short_description = _('Export all labels')
