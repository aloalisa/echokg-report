from django.shortcuts import render
from django import views

from .ml_algo import get_answer_for_test


class EchocardiographyReportView(views.View):
    template_name = 'index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        form_data = request.POST
        test_data = [
            form_data['aortic_diameter'],
            form_data['left_atrium'],
            form_data['left_ventricular_at_rest'],
            form_data['left_ventricular_size_during_contraction'],
            form_data['left_ventricular_mass_at_rest'],
            form_data['left_ventricular_mass_during_contraction'],
            form_data['left_ventricular_ejection_fraction'],
            form_data['intraventricular_septum'],
            form_data['posterior_wall_of_the_left_ventricle'],
            form_data['right_ventricle'],
        ]
        test_data = list(map(lambda s: float(s.replace(',', '.')), test_data))
        results = get_answer_for_test([test_data])
        print(results)
        return render(request, self.template_name, {
            'form_data': form_data,
            'results': results,
        })
