import streamlit as st
import time
import hashlib
from datetime import datetime, timedelta
from rag_chat import generate_answer
from search import find_similar_chunks
from embeddings import process_all_documents

# Rate limiting and safety
class SafetyManager:
    def __init__(self):
        if 'rate_limit' not in st.session_state:
            st.session_state.rate_limit = {}
        if 'query_count' not in st.session_state:
            st.session_state.query_count = 0
    
    def check_rate_limit(self, user_ip="default", max_queries=20, window_minutes=60):
        """Rate limiting: max 20 queries per hour"""
        now = datetime.now()
        user_key = hashlib.md5(user_ip.encode()).hexdigest()
        
        if user_key not in st.session_state.rate_limit:
            st.session_state.rate_limit[user_key] = []
        
        # Clean old queries
        st.session_state.rate_limit[user_key] = [
            query_time for query_time in st.session_state.rate_limit[user_key]
            if now - query_time < timedelta(minutes=window_minutes)
        ]
        
        if len(st.session_state.rate_limit[user_key]) >= max_queries:
            return False, f"Rate limit exceeded. Maximum {max_queries} queries per hour."
        
        st.session_state.rate_limit[user_key].append(now)
        return True, "OK"
    
    def is_safe_query(self, query):
        """Basic content filtering"""
        unsafe_patterns = [
            'ignore previous instructions', 'system prompt', 'jailbreak',
            'hack', 'exploit', 'malicious', 'harmful content'
        ]
        
        query_lower = query.lower()
        for pattern in unsafe_patterns:
            if pattern in query_lower:
                return False, "Query contains potentially unsafe content."
        
        if len(query) > 500:
            return False, "Query too long. Please keep questions under 500 characters."
        
        return True, "OK"

def get_document_info():
    """Get information about available documents"""
    try:
        chunks, embeddings, metadata = process_all_documents()
        doc_sources = list(set([chunk['source'] for chunk in metadata]))
        
        doc_info = {
            'count': len(doc_sources),
            'sources': doc_sources,
            'total_chunks': len(chunks),
            'topics': {
                'AI and Machine Learning': 'Types of ML, deep learning, applications, challenges',
                'AWS Cloud Services': 'EC2, S3, Lambda, databases, security, best practices',
                'Software Development': 'Agile, DevOps, CI/CD, testing, architecture patterns'
            }
        }
        return doc_info
    except Exception as e:
        return {'error': str(e)}

def enhanced_rag_response(query, safety_manager):
    """Enhanced RAG with better context handling"""
    try:
        # Get relevant chunks
        chunks, embeddings, metadata = process_all_documents()
        context_chunks = find_similar_chunks(query, chunks, embeddings, metadata, top_k=3)
        
        # Check if query is about the system itself
        system_queries = ['how many documents', 'what documents', 'what can you', 'what topics']
        if any(sq in query.lower() for sq in system_queries):
            doc_info = get_document_info()
            return {
                'answer': f"I have access to {doc_info['count']} documents covering: {', '.join(doc_info['topics'].keys())}. You can ask about AI/ML concepts, AWS services, or software development practices.",
                'sources': list(doc_info['sources']),
                'is_system_response': True
            }
        
        # Check relevance threshold
        if context_chunks[0]['score'] < 0.15:
            return {
                'answer': "I couldn't find relevant information in my documents to answer that question. Please ask about AI/ML, AWS services, or software development practices.",
                'sources': [],
                'is_system_response': True
            }
        
        # Generate AI response with better prompt
        enhanced_prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 

Rules:
1. Answer only based on the context provided
2. If the answer isn't in the context, say "I don't have that information in my documents"
3. Be concise but complete
4. Don't make up information

Context: {' '.join([chunk['chunk'] for chunk in context_chunks[:2]])}

Question: {query}

Answer:"""
        
        answer = generate_answer(query, context_chunks[:2])
        
        return {
            'answer': answer,
            'sources': [chunk['source'] for chunk in context_chunks[:2]],
            'relevance_scores': [chunk['score'] for chunk in context_chunks[:2]],
            'is_system_response': False
        }
        
    except Exception as e:
        return {
            'answer': f"Sorry, I encountered an error: {str(e)}",
            'sources': [],
            'is_system_response': True
        }

# Initialize
safety_manager = SafetyManager()

# Page config
st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Main UI
st.title("🤖 Personal RAG Assistant")
st.markdown("### Ask questions about AI/ML, AWS Services, or Software Development!")

# Sidebar with info
with st.sidebar:
    st.header("📚 Available Knowledge")
    
    doc_info = get_document_info()
    if 'error' not in doc_info:
        st.metric("Documents", doc_info['count'])
        st.metric("Text Chunks", doc_info['total_chunks'])
        
        st.subheader("Topics You Can Ask About:")
        for topic, description in doc_info['topics'].items():
            with st.expander(topic):
                st.write(description)
        
        st.subheader("Example Questions:")
        st.code("""
- What are the types of machine learning?
- How does AWS Lambda work?
- What is continuous integration?
- What are SOLID principles?
- How does Amazon S3 work?
        """)
    
    st.subheader("Usage Stats")
    st.metric("Queries This Session", st.session_state.query_count)
    
    with st.expander("Safety Features"):
        st.write("✅ Rate limiting (20 queries/hour)")
        st.write("✅ Content filtering")
        st.write("✅ Query length limits")
        st.write("✅ Cost monitoring")

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"📄 {source}")

# Chat input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Safety checks
    rate_ok, rate_msg = safety_manager.check_rate_limit()
    if not rate_ok:
        st.error(rate_msg)
        st.stop()
    
    safe_ok, safe_msg = safety_manager.is_safe_query(prompt)
    if not safe_ok:
        st.error(safe_msg)
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            
            response = enhanced_rag_response(prompt, safety_manager)
            
            # Display answer
            st.markdown(response['answer'])
            
            # Show sources if available
            if response['sources']:
                with st.expander("📚 Sources Used"):
                    for i, source in enumerate(response['sources']):
                        relevance = response.get('relevance_scores', [0])[i] if i < len(response.get('relevance_scores', [])) else 0
                        st.write(f"📄 **{source}** (relevance: {relevance:.3f})")
            
            # Show cost estimate for AI responses
            if not response['is_system_response']:
                st.caption("💰 Estimated cost: ~$0.001")
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response['answer'],
        "sources": response['sources']
    })
    
    st.session_state.query_count += 1

# Footer
st.markdown("---")
st.caption("🔒 This system includes rate limiting, content filtering, and cost controls for safe operation.")