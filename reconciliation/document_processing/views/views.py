from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import logging
from document_processing.utils.processors.invoice_processors.invoice_pdf_processor import InvoicePDFProcessor

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class ProcessInvoiceAPI(View):
    """
    API endpoint to process invoice PDF files
    """
    
    def post(self, request):
        """
        Process uploaded PDF invoice file
        
        Expected: multipart/form-data with 'pdf_file' field
        Returns: JSON response with extracted invoice data
        """
        try:
            # Check if file is provided
            if 'pdf_file' not in request.FILES:
                return JsonResponse({
                    'success': False,
                    'error': 'No PDF file provided. Please upload a file with key "pdf_file".',
                    'status': 'error'
                }, status=400)
            
            pdf_file = request.FILES['pdf_file']
            
            # Validate file type
            if not pdf_file.name.lower().endswith('.pdf'):
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid file type. Please upload a PDF file.',
                    'status': 'error'
                }, status=400)
            
            # Validate file size
            max_size = 10 * 1024 * 1024  # 10MB
            if pdf_file.size > max_size:
                return JsonResponse({
                    'success': False,
                    'error': f'File too large. Maximum size allowed is {max_size // (1024*1024)}MB.',
                    'status': 'error'
                }, status=400)
            
            # Initialize processor and process the file
            logger.info(f"Processing invoice file: {pdf_file.name}")
            processor = InvoicePDFProcessor()
            extracted_data = processor.process_uploaded_file(pdf_file)
            
            # Return successful response
            return JsonResponse({
                'success': True,
                'message': 'Invoice processed successfully',
                'status': 'completed',
                'data': extracted_data
            }, status=200)
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            return JsonResponse({
                'success': False,
                'error': str(ve),
                'status': 'validation_error'
            }, status=400)
            
        except Exception as e:
            logger.error(f"Error processing invoice: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to process invoice: {str(e)}',
                'status': 'processing_error'
            }, status=500)

