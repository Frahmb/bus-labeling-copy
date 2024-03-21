from django.shortcuts import render
from django.views import View
from django.utils import timezone
from django.http import HttpResponse, HttpResponseRedirect
from .models import Competition, CompetitionSubmission
from .forms import CompetitionSubmissionForm


class CompetitionView(View):
    def get(self, request, index):
        competition = Competition.objects.get(id=index)

        if request.user.is_authenticated:
            my_sub = CompetitionSubmission.objects.filter(
                creator=request.user, competition=competition
            ).order_by('-submit_time')
        else:
            my_sub = None

        all_sub = CompetitionSubmission.objects.filter(
            competition=competition
        ).order_by('-score')[:20]

        # if end_date is not set, then it is always valid
        valid = timezone.now() <= competition.end_date \
            if competition.end_date else True

        return render(
            request,
            'competition/competition.html',
            {'competition': competition,
             'valid': valid,
             'my_sub': my_sub,
             'all_sub': all_sub}
        )

    def post(self, request, index):
        if not request.user.is_authenticated:
            return HttpResponseRedirect(request.path)

        competition = Competition.objects.get(id=index)
        submission = CompetitionSubmission(
            competition=competition,
            creator=request.user,
        )

        form = CompetitionSubmissionForm(request.POST,
                                         request.FILES,
                                         instance=submission)

        if not form.is_valid():
            # return error message
            return HttpResponse('Invalid form')

        if competition.evaluation_func:
            # exec score function
            local_dict = {}
            exec(competition.evaluation_func, local_dict)
            local_dict['score'](competition, submission)

        submission.save()
        return HttpResponseRedirect(request.path)
