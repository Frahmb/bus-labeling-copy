import re
from django.utils.translation import gettext_lazy as _
from django import forms
from django.contrib.auth.models import User
from django.core.validators import validate_email
from .countries import COUNTRIES


class SignUpForm(forms.Form):
    username = forms.CharField(label=_('Username'), max_length=50)
    email = forms.EmailField(label=_('Email'))
    firstname = forms.CharField(label=_('First Name'), max_length=20)
    lastname = forms.CharField(label=_('Last Name'), max_length=20)
    organization = forms.CharField(label=_('Organization/Institution'),
                                   max_length=50)
    country = forms.ChoiceField(label=_("Country"), choices=COUNTRIES)
    password1 = forms.CharField(label=_('Password'), widget=forms.PasswordInput)
    password2 = forms.CharField(label=_('Password Confirmation'),
                                widget=forms.PasswordInput)

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if len(username) < 3:
            raise forms.ValidationError(
                _("Your username must be at least 3 characters long"))
        elif len(username) > 20:
            raise forms.ValidationError(
                _("Your username is too long. The max length is 20"))
        else:
            filter_result = User.objects.filter(username__exact=username)
            if len(filter_result) > 0:
                raise forms.ValidationError(_('Your username already exists'))
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        try :
            validate_email(email)
            filter_result = User.objects.filter(email__exact=email)
            if len(filter_result) > 0:
                raise forms.ValidationError(_("Your email already exists"))
        except:
            raise forms.ValidationError(_("Please enter a valid email"))

        return email

    def clean_firstname(self):
        name = self.cleaned_data.get('firstname')
        if not name:
            raise forms.ValidationError(
                _("The first name cannot be empty.")
            )
        return name

    def clean_lastname(self):
        name = self.cleaned_data.get('lastname')
        if not name:
            raise forms.ValidationError(
                _("The last name cannot be empty.")
            )
        return name

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        if len(password1) < 3:
            raise forms.ValidationError(_("Your password is too short"))
        elif len(password1) > 20:
            raise forms.ValidationError(_("Your password is too long"))

        return password1

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise forms.ValidationError(
                _('Password mismatch Please enter again'))

        return password2

    def clean_organization(self):
        organization = self.cleaned_data.get('organization', '')
        if len(organization) == 0:
            raise forms.ValidationError(
                _("Organization/institution cannot be empty"))
        return organization

    def clean_country(self):
        country = self.cleaned_data.get('country')
        if not country:
            raise forms.ValidationError(_("Country filed is required"))
        return country
