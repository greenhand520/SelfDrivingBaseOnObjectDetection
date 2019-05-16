# -*- coding: utf-8 -*-
# @author: mdmbct
# @date:   5/1/2019 7:02 PM
# @last modified by: 
# @last modified time: 5/1/2019 7:02 PM
from django.shortcuts import render
from .model import *
from django.http import StreamingHttpResponse
from .settings import STATICFILES_DIRS
from django.http import FileResponse

def index(request):
    context = {}
    # print(STATICFILES_DIRS[0] + "/source/data_set/")
    data_sets = get_all_data_sets(STATICFILES_DIRS[0] + "/source/data_set/")
    models = get_all_model(STATICFILES_DIRS[0] + "/source/model/")
    context = {"datas": data_sets, "models": models}
    return render(request, 'self_driving_data_set/index.html', context)

def download(request):
    file_name = request.GET.get("file_name")
    # print(file_name)
    saved_name = file_name
    source = file_name.split('_')[0]
    if source == 'set':
        file_name = STATICFILES_DIRS[0] + "/source/data_set/" + file_name
    elif source == 'model':
        file_name = STATICFILES_DIRS[1] + "/source/model/" + file_name

    file = open(file_name, 'rb')
    response = FileResponse(file)
    response['Content-Type'] = 'application/octet-stream'
    response['Content-Disposition'] = 'attachment;filename=' + saved_name
    return response
