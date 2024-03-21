import os
from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponse, HttpResponseRedirect
from django.views import View
from django.apps import apps
from .models import Masking


class MaskingView(View):
    def get(self, request, masking_type, ds_name, obj_id=None):
        model = apps.all_models['labeling'].get(ds_name)
        if model is None:
            return HttpResponse(status=404)

        if obj_id is None:
            img = model.get_next_unlabeled(request.user, masking_type)
            return HttpResponseRedirect(
                reverse('masking_change',
                        args=(masking_type, ds_name, img.id)))
        else:
            img = model.objects.get(id=obj_id)

        next_img = model.get_next(img)
        prev_img = model.get_prev(img)

        next_url = reverse(
            'masking_change',
            args=(masking_type, ds_name, next_img.id)
        ) if next_img else ''
        prev_url = reverse(
            'masking_change',
            args=(masking_type, ds_name, prev_img.id)
        ) if prev_img else ''

        ds_url = reverse(
            f'labeling:labeling_{img.case._meta.model_name}_changelist')
        mask = img.get_mask(request.user)

        case_url = reverse(
            f'labeling:labeling_{img.case._meta.model_name}_change',
            args=(img.case.id,)
        )

        return render(request,
                      f'labeling/masking_{masking_type}.html',
                      {'img_url': img.image.url,
                       'img': img,
                       'mask': mask,
                       'img_name': os.path.basename(img.image.name),
                       'next_url': next_url,
                       'prev_url': prev_url,
                       'case_url': case_url,
                       'ds_url': ds_url,
                       'ds_name': ds_name,
                       'description': img.description()})

    def post(self, request, masking_type, ds_name, obj_id):
        try:
            masking = Masking.objects.filter(id=request.POST['id']).first()
        except:
            return HttpResponse(status=404)

        setattr(masking, masking_type, request.POST['data'])
        masking.save()
        return HttpResponse(status=200)
