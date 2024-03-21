from django.shortcuts import render
from django.conf import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.views import View
from django.core.mail import EmailMultiAlternatives
from django.http import HttpResponse
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.contrib.auth.tokens import default_token_generator
from django.contrib.auth import login
from django.contrib.auth.models import User, Group
from labeling.models import BUSDataset
from .forms import SignUpForm


class SignUpView(View):
    def get(self, request):
        form = SignUpForm()
        return render(request, 'admin/register.html',
                      {'form': form, 'title': 'Sign Up'})

    def post(self, request):
        form = SignUpForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            firstname = form.cleaned_data['firstname']
            lastname = form.cleaned_data['lastname']
            password = form.cleaned_data['password2']
            organization = form.cleaned_data['organization']
            country = form.cleaned_data['country']

            user = User.objects.create_user(username=username,
                                            password=password,
                                            email=email,
                                            first_name=firstname,
                                            last_name=lastname)
            user.is_staff = True
            user.is_active = False
            user.profile.organization = organization
            user.profile.country = country
            user.profile.save()
            user.save()

            # add new user to all groups
            for group_name in BUSDataset.GROUPS:
                user.groups.add(Group.objects.get(name=group_name))

            current_site = get_current_site(request)

            mail_subject = 'Activate your account from MIDA Labeling System'
            mail_body = render_to_string(
                'registration/registration_confirmation.html',
                {
                    'user': user,
                    'domain': current_site.domain,
                    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
                    'token': default_token_generator.make_token(user)
                },
                request
            )
            mail = EmailMultiAlternatives(
                mail_subject,
                mail_body,
                from_email=settings.EMAIL_FROM,
                to=[email]
            )
            mail.send()
            return render(request, 'registration/registration_done.html')
        else:
            return render(request, 'admin/register.html', {'form': form})


def activate(request, uidb64, token):
    try:
        uid = force_text(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        login(request, user)
        return render(request, 'registration/activation_done.html')
    else:
        return HttpResponse('Activation link is invalid!')
