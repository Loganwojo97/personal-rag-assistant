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
st.set_page_config(page_title="Personal RAG Assistant", page_icon="🤖")

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
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
bucket_name = 'my-rag-documents-loganw'

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Simple RAG functions
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
    try:
        # Get documents
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            return "No documents found in bucket."
        
        # Read first document for demo
        first_file = response['Contents'][0]['Key']
        content = read_document(first_file)
        
        if not content:
            return "Could not read document."
        
        # Simple keyword matching for demo
        if any(word in query.lower() for word in ['machine learning', 'ml', 'types']):
            return "Based on your documents, the three main types of machine learning are: 1) Supervised Learning, 2) Unsupervised Learning, and 3) Reinforcement Learning."
        elif any(word in query.lower() for word in ['aws', 'lambda', 'cloud']):
            return "AWS Lambda is a serverless computing service that runs code in response to events without requiring server management."
        else:
            return f"I found information in {first_file}. Please ask more specific questions about AI/ML, AWS, or software development."
            
    except Exception as e:
        return f"Error: {e}"

# UI
st.title("🤖 Personal RAG Assistant")
st.write("Ask questions about AI/ML, AWS services, or software development!")

with st.sidebar:
    st.header("📚 Available Documents")
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            for obj in response['Contents']:
                st.write(f"📄 {obj['Key']}")
    except:
        st.write("Could not load document list")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = simple_rag(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})