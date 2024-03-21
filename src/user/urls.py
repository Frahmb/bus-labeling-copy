from django.urls import path
from django.conf import settings
from django.contrib.auth import views as auth_views
from .views import SignUpView, activate


urlpatterns = [
    path('signup', SignUpView.as_view(), name='signup'),
    path(
        'labeling/password_reset/',
        auth_views.PasswordResetView.as_view(from_email=settings.EMAIL_FROM),
        name='admin_password_reset',
    ),
    path(
        'labeling/password_reset/done/',
        auth_views.PasswordResetDoneView.as_view(),
        name='password_reset_done',
    ),
    path(
        'reset/<uidb64>/<token>/',
        auth_views.PasswordResetConfirmView.as_view(),
        name='password_reset_confirm',
    ),
    path(
        'reset/done/',
        auth_views.PasswordResetCompleteView.as_view(),
        name='password_reset_complete',
    ),
    path(
        'activation/<uidb64>/<token>/',
        activate,
        name='activation'
    )
]
