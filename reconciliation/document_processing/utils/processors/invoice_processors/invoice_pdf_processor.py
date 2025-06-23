import fitz
import json
import logging
import tempfile
import os
from typing import Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from django.conf import settings
from datetime import datetime
import tiktoken

logger = logging.getLogger(__name__)

class InvoicePDFProcessor:
    """
    Invoice PDF processor using LangChain and Google Generative AI
    """
    
    def __init__(self):
        """Initialize the processor"""
        self.api_key = getattr(settings, 'GOOGLE_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set in Django settings or environment variables")
        
        self.model_name = getattr(settings, 'GEMINI_MODEL', 'gemini-1.5-flash')
        
        self.llm = GoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.1  
        )

        try:
            self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        except:
            # Fallback to a basic encoder if gpt-4 not available
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
        # Define the invoice schema
        self.invoice_schema = {
            "vendor_name": "",
            "vendor_pan": "",
            "vendor_gst": "",
            "invoice_date": "",
            "invoice_number": "",
            "po_number": "",
            "items": [
                {
                    "description": "",
                    "hsn_code": "",
                    "quantity": "",
                    "unit_of_measurement": "",
                    "invoice_value_item_wise": ""
                }
            ],
            "invoice_value_without_gst": "",
            "gst_details": {
                "cgst_rate": "",
                "cgst_amount": "",
                "sgst_rate": "",
                "sgst_amount": "",
                "igst_rate": "",
                "igst_amount": "",
                "total_gst_amount": ""
            },
            "invoice_total_post_gst": ""
        }
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Input text to count tokens
            
        Returns:
            Number of tokens
        """
        try:
            return len(self.token_encoder.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using word count approximation")
            # Fallback: approximate tokens as words * 1.3
            return int(len(text.split()) * 1.3)
    
    def extract_pan_from_gst(self, gst_number: str) -> str:
        """
        Extract PAN from GST number by removing first 2 and last 3 characters
        
        Args:
            gst_number: 15-digit GST number
            
        Returns:
            10-character PAN number
        """
        if not gst_number or len(gst_number) < 15:
            return ""
        
        # Remove first 2 characters (state code) and last 3 characters (checksum + entity code)
        pan = gst_number[2:12]  # Extract characters 3-12 (0-indexed: 2-11)
        return pan
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def create_extraction_prompt(self) -> PromptTemplate:
        """
        Create prompt template for invoice extraction
        
        Returns:
            PromptTemplate for invoice extraction
        """
        template = """
You are an expert invoice data extraction system. Extract structured information from the following invoice text and return it as a valid JSON object.

EXTRACTION RULES:
1. Extract ALL available information from the invoice
2. If a field is not found or unclear, use an empty string ""
3. For numerical values, extract only the number (remove currency symbols like ₹, $, etc.)
4. For dates, use DD/MM/YYYY or DD-MM-YYYY format
5. For GST numbers, extract the complete 15-digit alphanumeric code
6. For PAN numbers, extract the 10-character alphanumeric code
7. For items array, include ALL line items found in the invoice
8. For GST details, extract CGST, SGST, IGST rates and amounts as provided in the invoice. If specific breakup is not available, use empty strings.
9. Unit of measurement should be units like KG, PCS, NOS, LTR, etc.
10. HSN codes should be extracted for each item - HSN codes can be 4, 6, or 8 digits. Extract the FULL HSN code as shown in the invoice.
11. **IMPORTANT - PO vs Order Number**: 
    - Extract PO number from "PO Number", "Reference PO", "PO No" fields ONLY
    - Do NOT use "Order Number" or "Order No" as PO number
    - If no explicit PO reference found, use empty string for po_number
12. Be precise and accurate - double-check all extracted values
13. Return ONLY the JSON object, no additional text

REQUIRED JSON STRUCTURE:
{schema}

INVOICE TEXT TO PROCESS:
{invoice_text}

Extract the information and return the JSON object:
"""
        
        return PromptTemplate(
            input_variables=["schema", "invoice_text"],
            template=template
        )
    
    def validate_and_clean_json(self, json_str: str) -> Dict[str, Any]:
        """
        Validate and clean the extracted JSON response
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Cleaned and validated JSON dict
        """
        try:
            # Clean the response - remove markdown formatting if present
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            # Parse the JSON
            extracted_data = json.loads(json_str)
            
            # Validate structure against schema
            validated_data = self.invoice_schema.copy()
            
            # Update validated_data with extracted_data, preserving structure
            for key, value in extracted_data.items():
                if key in validated_data:
                    if isinstance(validated_data[key], dict) and isinstance(value, dict):
                        # For nested dictionaries, update individual keys
                        for sub_key, sub_value in value.items():
                            if sub_key in validated_data[key]:
                                validated_data[key][sub_key] = sub_value
                    else:
                        validated_data[key] = value
            
            # Auto-extract PAN from GST if PAN is not provided
            if validated_data.get("vendor_gst") and not validated_data.get("vendor_pan"):
                extracted_pan = self.extract_pan_from_gst(validated_data["vendor_gst"])
                if extracted_pan:
                    validated_data["vendor_pan"] = extracted_pan
                    logger.info(f"Extracted PAN {extracted_pan} from GST number")
            
            return validated_data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Raw JSON string: {json_str[:500]}...")
            raise ValueError(f"Invalid JSON format returned by LLM: {str(e)}")
        except Exception as e:
            logger.error(f"Error validating JSON: {str(e)}")
            raise
    
    def process_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Process Django uploaded file and extract invoice data
        
        Args:
            uploaded_file: Django UploadedFile object
            
        Returns:
            Dictionary containing extracted invoice data
        """
        temp_path = None
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name
            
            # Extract text from PDF
            logger.info(f"Extracting text from uploaded file: {uploaded_file.name}")
            extracted_text = self.extract_text_from_pdf(temp_path)
            
            if not extracted_text:
                raise ValueError("No text could be extracted from the PDF file")
            
            # Create and format the prompt
            prompt_template = self.create_extraction_prompt()
            formatted_prompt = prompt_template.format(
                schema=json.dumps(self.invoice_schema, indent=2),
                invoice_text=extracted_text
            )

            # Count input tokens
            input_tokens = self.count_tokens(formatted_prompt)
            logger.info(f"Input tokens: {input_tokens}")
            
            # Get response from LLM
            logger.info("Processing invoice data with LLM...")
            llm_response = self.llm.invoke(formatted_prompt)

            # Count output tokens
            output_tokens = self.count_tokens(llm_response)
            total_tokens = input_tokens + output_tokens
            logger.info(f"Output tokens: {output_tokens}, Total tokens: {total_tokens}")
            
            # Validate and clean the JSON response
            extracted_data = self.validate_and_clean_json(llm_response)
            
            # Add processing metadata
            extracted_data["_metadata"] = {
                "filename": uploaded_file.name,
                "file_size": uploaded_file.size,
                "processed_at": datetime.now().isoformat(),
                "model_used": self.model_name,
                "text_length": len(extracted_text),
                "processing_status": "success",
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            logger.info(f"Successfully processed invoice: {uploaded_file.name}")
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing uploaded file {uploaded_file.name}: {str(e)}")
            
            # Return error information
            error_data = self.invoice_schema.copy()
            error_data["_metadata"] = {
                "filename": uploaded_file.name,
                "file_size": getattr(uploaded_file, 'size', 0),
                "processed_at": datetime.now().isoformat(),
                "processing_status": "failed",
                "error_message": str(e),
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            raise Exception(f"Failed to process invoice: {str(e)}")
            
        
    
    def process_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Process invoice from file path (for testing purposes)
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted invoice data
        """
        try:
            # Extract text from PDF
            logger.info(f"Extracting text from file: {file_path}")
            extracted_text = self.extract_text_from_pdf(file_path)
            
            if not extracted_text:
                raise ValueError("No text could be extracted from the PDF file")
            
            # Create and format the prompt
            prompt_template = self.create_extraction_prompt()
            formatted_prompt = prompt_template.format(
                schema=json.dumps(self.invoice_schema, indent=2),
                invoice_text=extracted_text
            )
            
            # Count input tokens
            input_tokens = self.count_tokens(formatted_prompt)
            logger.info(f"Input tokens: {input_tokens}")

            # Get response from LLM
            logger.info("Processing invoice data with LLM...")
            llm_response = self.llm.invoke(formatted_prompt)

            # Count output tokens
            output_tokens = self.count_tokens(llm_response)
            total_tokens = input_tokens + output_tokens
            logger.info(f"Output tokens: {output_tokens}, Total tokens: {total_tokens}")
            
            # Validate and clean the JSON response
            extracted_data = self.validate_and_clean_json(llm_response)
            
            # Add processing metadata
            extracted_data["_metadata"] = {
                "file_path": file_path,
                "processed_at": datetime.now().isoformat(),
                "model_used": self.model_name,
                "text_length": len(extracted_text),
                "processing_status": "success",
                "token_usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
            }
            
            logger.info(f"Successfully processed invoice from: {file_path}")
            logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise Exception(f"Failed to process invoice: {str(e)}")


