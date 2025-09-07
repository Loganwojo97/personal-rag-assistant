import boto3
import PyPDF2
from io import BytesIO

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'my-rag-documents-loganw'

def list_documents():
    """List all documents in the S3 bucket"""
    response = s3.list_objects_v2(Bucket=bucket_name)
    if 'Contents' in response:
        files = [obj['Key'] for obj in response['Contents']]
        print(f"Found {len(files)} documents:")
        for file in files:
            print(f"  - {file}")
        return files
    else:
        print("No documents found in bucket")
        return []

def read_document(file_key):
    """Read and extract text from a document"""
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        
        # Handle different file types
        if file_key.lower().endswith('.pdf'):
            # Read PDF content
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif file_key.lower().endswith(('.txt', '.md')):
            text = file_content.decode('utf-8')
        else:
            print(f"Unsupported file type: {file_key}")
            return None
            
        print(f"\n--- Content from {file_key} ---")
        print(f"Length: {len(text)} characters")
        print(f"First 200 characters:\n{text[:200]}...")
        
        return text
        
    except Exception as e:
        print(f"Error reading {file_key}: {str(e)}")
        return None

if __name__ == "__main__":
    print("Testing S3 connection...")
    files = list_documents()
    
    if files:
        # Read the first document
        first_file = files[0]
        content = read_document(first_file)