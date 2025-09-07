import boto3
import json
from search import find_similar_chunks
from embeddings import process_all_documents

# Initialize Bedrock client
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def generate_answer(query, context_chunks):
    """Generate answer using AWS Bedrock with context"""
    
    # Combine context chunks
    context = "\n\n".join([chunk['chunk'] for chunk in context_chunks])
    
    # Create prompt
    prompt = f"""Based on the following context, please answer the question. If the answer isn't in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    # Call Bedrock (Claude)
    body = json.dumps({
        "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.1,
        "top_p": 0.9,
    })
    
    try:
        response = bedrock.invoke_model(
            body=body,
            modelId='anthropic.claude-instant-v1',
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion')
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def chat_with_documents(query):
    """Main RAG chat function"""
    print(f"Question: {query}")
    print("Searching documents...")
    
    # Get document chunks and embeddings
    chunks, embeddings, metadata = process_all_documents()
    
    # Find relevant context
    context_chunks = find_similar_chunks(query, chunks, embeddings, metadata, top_k=2)
    
    print("Generating answer...")
    answer = generate_answer(query, context_chunks)
    
    print(f"\nAnswer: {answer}")
    print(f"\nSources used:")
    for chunk in context_chunks:
        print(f"- {chunk['source']} (score: {chunk['score']:.3f})")
    
    return answer

if __name__ == "__main__":
    test_question = "What are the three types of machine learning?"
    chat_with_documents(test_question)