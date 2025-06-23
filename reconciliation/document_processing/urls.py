from django.urls import path
from .views import views

app_name = 'document_processing'

urlpatterns = [
    path('api/process-invoice/', views.ProcessInvoiceAPI.as_view(), name='process_invoice'),
    
]