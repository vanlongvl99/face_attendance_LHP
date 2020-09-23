# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

# record = {  "first_name": "long",
            # "last_name":"nguyen",
            # "education":"bk",
            # "occupation":"student",
            # "residence":"residence",
            # "country":"vietnam",
            # "marital_status":"single",
            # "recorded_at":"recorded_at"}

def index(request):
    records = Records.objects.all()[:10]    #getting the first 10 records
    print("\nrecord:",records)
    context = {
        'records': records
    }
    return render(request, 'records.html', context)

def details(request, id):
    print("\n id:",id)
    record = Records.objects.get(id=id)
    print("record id:", record)
    context = {
        'record' : record
    }
    return render(request, 'details.html', context)
