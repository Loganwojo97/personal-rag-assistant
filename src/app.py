import streamlit as st
from rag_chat import chat_with_documents
from search import find_similar_chunks
from embeddings import process_all_documents

st.title("🤖 Personal RAG Assistant")
st.write("Ask questions about your documents!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            
            # Get answer (this costs ~$0.001 per question)
            chunks, embeddings, metadata = process_all_documents()
            context_chunks = find_similar_chunks(prompt, chunks, embeddings, metadata, top_k=2)
            
            # For demo, let's show the search results
            st.write("**Found relevant information in:**")
            for chunk in context_chunks:
                st.write(f"📄 {chunk['source']} (relevance: {chunk['score']:.3f})")
            
            # You can uncomment this to get AI responses (costs money)
            # answer = chat_with_documents(prompt)
            # st.markdown(answer)
            
            # For now, just show the relevant chunks
            st.write("**Most relevant content:**")
            st.write(context_chunks[0]['chunk'][:500] + "...")
    
    st.session_state.messages.append({"role": "assistant", "content": "Response generated!"})