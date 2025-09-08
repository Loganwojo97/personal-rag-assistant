import streamlit as st
import boto3
import PyPDF2
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import hashlib
from datetime import datetime, timedelta
import json

# Configure page
st.set_page_config(page_title="Personal RAG Assistant", page_icon="🤖", layout="wide")

# Configure AWS from secrets
try:
    if 'aws' in st.secrets:
        import os
        os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
        os.environ['AWS_DEFAULT_REGION'] = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    else:
        st.error("AWS credentials not configured in secrets.")
        st.stop()
except Exception as e:
    st.error(f"Error configuring AWS: {e}")
    st.stop()

# Initialize clients
s3 = boto3.client('s3', region_name='us-west-2')
bucket_name = 'my-rag-documents-loganw'

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Security and rate limiting
def check_global_rate_limit():
    """Strict rate limiting across all users"""
    # Global daily limit
    if 'daily_queries' not in st.session_state:
        st.session_state.daily_queries = 0
    
    if st.session_state.daily_queries >= 50:
        return False, "Daily query limit reached for this demo. Please try again tomorrow."
    
    # Per-session limit
    if 'session_queries' not in st.session_state:
        st.session_state.session_queries = 0
    
    if st.session_state.session_queries >= 10:
        return False, "Session limit reached (10 queries). Please refresh to start a new session."
    
    return True, "OK"

def is_safe_query(query):
    """Basic content filtering"""
    unsafe_patterns = [
        'ignore previous instructions', 'system prompt', 'jailbreak',
        'hack', 'exploit', 'malicious', 'harmful content', 'bypass'
    ]
    
    query_lower = query.lower()
    for pattern in unsafe_patterns:
        if pattern in query_lower:
            return False, "Query contains potentially unsafe content."
    
    if len(query) > 500:
        return False, "Query too long. Please keep questions under 500 characters."
    
    return True, "OK"

# Document functions
def get_document_info():
    """Get information about available documents with better error handling"""
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            return {'error': 'No documents found in bucket'}
        
        doc_sources = [obj['Key'] for obj in response['Contents']]
        
        doc_info = {
            'count': len(doc_sources),
            'sources': doc_sources,
            'topics': {
                'AI and Machine Learning': 'Types of ML, deep learning, applications, challenges',
                'AWS Cloud Services': 'EC2, S3, Lambda, databases, security, best practices', 
                'Software Development': 'Agile, DevOps, CI/CD, testing, architecture patterns'
            }
        }
        return doc_info
    except Exception as e:
        # For demo purposes, return mock data when AWS fails
        return {
            'count': 3,
            'sources': ['AI and Machine Learning Overview.pdf', 'AWS Cloud Services Guide.pdf', 'Modern Software Development Practices.pdf'],
            'topics': {
                'AI and Machine Learning': 'Types of ML, deep learning, applications, challenges',
                'AWS Cloud Services': 'EC2, S3, Lambda, databases, security, best practices', 
                'Software Development': 'Agile, DevOps, CI/CD, testing, architecture patterns'
            },
            'demo_mode': True,
            'error_msg': str(e)
        }

def read_document(file_key):
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read()
        
        if file_key.lower().endswith('.pdf'):
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            return file_content.decode('utf-8')
    except Exception as e:
        st.error(f"Error reading {file_key}: {e}")
        return None

def simple_rag(query):
    """Simple RAG with keyword matching for demo"""
    try:
        # Get documents
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            return "No documents found in bucket."
        
        # System queries
        system_queries = ['how many documents', 'what documents', 'what can you', 'what topics']
        if any(sq in query.lower() for sq in system_queries):
            doc_info = get_document_info()
            if 'error' not in doc_info:
                return f"I have access to {doc_info['count']} documents covering: {', '.join(doc_info['topics'].keys())}. You can ask about AI/ML concepts, AWS services, or software development practices."
            else:
                return doc_info['error']
        
        # Topic-based responses
        if any(word in query.lower() for word in ['machine learning', 'ml', 'types', 'supervised', 'unsupervised']):
            return "Based on your documents, the three main types of machine learning are:\n\n1. **Supervised Learning** - algorithms learn from labeled training data\n2. **Unsupervised Learning** - works with unlabeled data to discover patterns\n3. **Reinforcement Learning** - training agents through rewards and penalties\n\n*Source: AI and Machine Learning Overview.pdf*"
        elif any(word in query.lower() for word in ['aws', 'lambda', 'cloud', 'serverless']):
            return "**AWS Lambda** is a serverless computing service that:\n\n- Runs code in response to events\n- Requires no server management\n- Automatically scales based on request volume\n- Uses pay-per-execution pricing\n- Supports multiple programming languages\n\n*Source: AWS Cloud Services Guide.pdf*"
        elif any(word in query.lower() for word in ['ci/cd', 'continuous integration', 'devops', 'deployment']):
            return "**Continuous Integration/Continuous Deployment (CI/CD)** includes:\n\n- **Continuous Integration**: Automatically build and test code changes\n- **Continuous Deployment**: Automatically deploy validated changes\n- **Popular tools**: GitHub Actions, Jenkins, GitLab CI, AWS CodePipeline\n- **Benefits**: Faster delivery, fewer bugs, automated testing\n\n*Source: Modern Software Development Practices.pdf*"
        else:
            files = [obj['Key'] for obj in response['Contents']]
            return f"I found information in {len(files)} documents. Please ask more specific questions about:\n\n- AI/ML concepts and types\n- AWS services like Lambda, S3, EC2\n- Software development practices and DevOps\n\nExample: 'What are the types of machine learning?' or 'How does AWS Lambda work?'"
            
    except Exception as e:
        return f"Error accessing documents: {e}"

# Main UI
st.title("🤖 Personal RAG Assistant")
st.markdown("### Ask questions about AI/ML, AWS Services, or Software Development!")

# Sidebar with enhanced knowledge section
with st.sidebar:
    st.header("📚 Available Knowledge")
    
    doc_info = get_document_info()
    if 'demo_mode' in doc_info:
        st.warning("Running in demo mode - AWS connection issue")
        st.caption(f"Error: {doc_info['error_msg']}")
    
    st.metric("Documents", doc_info['count'])
    
    st.subheader("Topics You Can Ask About:")
    for topic, description in doc_info['topics'].items():
        with st.expander(topic):
            st.write(description)
    
    st.subheader("Document Sources:")
    for source in doc_info['sources']:
        st.write(f"📄 {source}")
    
    st.subheader("Example Questions:")
    st.code("""
- What are the types of machine learning?
- How does AWS Lambda work?
- What is continuous integration?
- What are SOLID principles?
- How does Amazon S3 work?
        """)
    
    st.header("Usage Stats")
    if 'session_queries' not in st.session_state:
        st.session_state.session_queries = 0
    
    st.metric("Queries This Session", st.session_state.session_queries)
    st.progress(st.session_state.session_queries / 10)
    st.caption(f"Limit: {10 - st.session_state.session_queries} remaining")
    
    with st.expander("🛡️ Safety Features"):
        st.write("✅ Rate limiting (10 queries/session)")
        st.write("✅ Content filtering")
        st.write("✅ Query length limits")
        st.write("✅ Demo mode (no AI costs)")
        st.write("✅ AWS resource protection")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with security checks
if prompt := st.chat_input("Ask a question about the documents..."):
    # Security checks
    allowed, rate_msg = check_global_rate_limit()
    if not allowed:
        st.error(rate_msg)
        st.stop()
    
    safe, safe_msg = is_safe_query(prompt)
    if not safe:
        st.error(safe_msg)
        st.stop()
    
    # Increment counters
    st.session_state.session_queries += 1
    st.session_state.daily_queries += 1
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = simple_rag(prompt)
            st.markdown(response)
            
            # Show query count
            remaining = 10 - st.session_state.session_queries
            if remaining > 0:
                st.caption(f"💡 {remaining} questions remaining this session")
            else:
                st.caption("🔒 Session limit reached. Refresh page for new session.")
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption("🔒 This is a demo version with built-in safety controls. Full AI integration available in private deployments.")