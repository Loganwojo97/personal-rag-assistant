# Personal RAG Assistant

A Retrieval-Augmented Generation (RAG) system that allows you to chat with your documents using AWS Bedrock and Claude.

## Features
- 📄 Document processing (PDF support)
- 🔍 Semantic search with embeddings
- 🤖 AI-powered responses using AWS Bedrock/Claude
- 🛡️ Built-in safety features (rate limiting, content filtering)
- 🌐 Web interface with Streamlit

## Architecture
- **Storage**: AWS S3 for document storage
- **Embeddings**: Local sentence-transformers model
- **AI**: AWS Bedrock with Claude
- **Interface**: Streamlit web app

## Demo
[Live Demo](your-streamlit-url-here)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure AWS credentials
4. Upload documents to S3
5. Run: `streamlit run src/enhanced_app.py`

## Cost
- Approximately $0.001-0.002 per question
- S3 storage: ~$0.02/month for sample documents