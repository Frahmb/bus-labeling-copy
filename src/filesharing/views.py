from django.utils import timezone
from django.http import FileResponse, HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import File


@login_required
def download(request, file_id):
    fail_reponse = HttpResponse(status=404, content='File not found or expired.')

    download_file = File.objects.get(id=file_id)

    # return 404 if file is not found or expired
    if not download_file:
        return fail_reponse
    
    if download_file.valid_until and download_file.valid_until < timezone.now():
        return fail_reponse

    # track download
    download_file.track_download(request.user)
    # return file
    return FileResponse(download_file.file, as_attachment=True)
