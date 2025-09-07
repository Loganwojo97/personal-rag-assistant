import streamlit as st
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure AWS credentials from Streamlit secrets
try:
    if st.secrets:
        os.environ['AWS_ACCESS_KEY_ID'] = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
        os.environ['AWS_SECRET_ACCESS_KEY'] = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"] 
        os.environ['AWS_DEFAULT_REGION'] = st.secrets["aws"]["AWS_DEFAULT_REGION"]
except:
    st.error("Please configure AWS credentials in Streamlit Cloud secrets.")
    st.info("Go to App settings > Secrets and add your AWS credentials")
    st.stop()

# Import and run the main app
from enhanced_app import *