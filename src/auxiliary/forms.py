from django import forms
from django.utils.translation import gettext_lazy as _
from .models import CompetitionSubmission


class CompetitionSubmissionForm(forms.ModelForm):
    class Meta:
        model = CompetitionSubmission
        fields = ['content', 'submit_file']
