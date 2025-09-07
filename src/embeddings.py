import boto3
from sentence_transformers import SentenceTransformer
from document_processor import read_document, list_documents
import numpy as np

# Initialize embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_embeddings(text_chunks):
    """Create embeddings for text chunks"""
    print(f"Creating embeddings for {len(text_chunks)} chunks...")
    embeddings = model.encode(text_chunks)
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings

def process_all_documents():
    """Process all documents and create embeddings"""
    files = list_documents()
    
    all_chunks = []
    all_embeddings = []
    chunk_metadata = []  # Store which document each chunk came from
    
    for file_key in files:
        print(f"\nProcessing {file_key}...")
        content = read_document(file_key)
        
        if content:
            chunks = chunk_text(content)
            embeddings = create_embeddings(chunks)
            
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)
            
            # Track metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'source': file_key,
                    'chunk_index': i,
                    'chunk_text': chunk[:100] + '...'  # Preview
                })
    
    print(f"\nTotal processed: {len(all_chunks)} chunks from {len(files)} documents")
    return all_chunks, np.array(all_embeddings), chunk_metadata

if __name__ == "__main__":
    chunks, embeddings, metadata = process_all_documents()
    
    # Show some results
    print(f"\nFirst chunk preview:")
    print(f"Source: {metadata[0]['source']}")
    print(f"Text: {chunks[0][:200]}...")
    print(f"Embedding shape: {embeddings[0].shape}")