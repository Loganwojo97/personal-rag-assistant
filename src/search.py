import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings import process_all_documents
import boto3

# Load the same model for consistency
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_chunks(query, chunks, embeddings, metadata, top_k=3):
    """Find most similar chunks to the query"""
    
    # Create embedding for the query
    query_embedding = model.encode([query])
    
    # Calculate similarity scores (cosine similarity)
    similarities = np.dot(embeddings, query_embedding.T).flatten()
    
    # Get top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'score': similarities[idx],
            'source': metadata[idx]['source'],
            'chunk_index': metadata[idx]['chunk_index']
        })
    
    return results

def search_documents(query):
    """Main search function"""
    print(f"Searching for: '{query}'")
    print("Loading documents and embeddings...")
    
    # Process all documents (in real app, you'd cache this)
    chunks, embeddings, metadata = process_all_documents()
    
    # Find similar chunks
    results = find_similar_chunks(query, chunks, embeddings, metadata)
    
    print(f"\nTop {len(results)} relevant chunks:")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result['source']}")
        print(f"   Similarity Score: {result['score']:.3f}")
        print(f"   Content: {result['chunk'][:300]}...")
        print("-" * 30)
    
    return results

if __name__ == "__main__":
    # Test some queries
    test_queries = [
        "What is machine learning?",
        "How does AWS Lambda work?",
        "What is continuous integration?"
    ]
    
    for query in test_queries:
        print("\n" + "="*60)
        search_documents(query)
        print("\n")