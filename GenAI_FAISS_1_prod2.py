import os
import boto3
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

# AWS Credentials (hardcoded for simplicity; consider using environment variables in production)
AWS_ACCESS_KEY = 'your_access_key'
AWS_SECRET_KEY = 'your_secret_key'
AWS_REGION = 'us-east-1'

# Initialize Boto3 Bedrock Client
bedrock_client = boto3.client(
    'bedrock',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Recursive Character Text Splitter
def recursive_character_splitter(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to get embeddings from Cohere via AWS Bedrock
def get_embeddings(text):
    response = bedrock_client.invoke_model(
        ModelId='cohere.embed-english-v3',
        ContentType='application/json',
        Body={"text": text}
    )
    embedding = np.array(response['Body']['embedding'])
    return embedding

# Function to create and populate FAISS index
def create_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to retrieve from FAISS Vector Database
def retrieve_from_faiss(query_embedding, index, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return distances, indices

# Function to store FAISS index in S3
def store_faiss_index_in_s3(index, bucket_name, file_name):
    faiss.write_index(index, file_name)
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    s3_client.upload_file(file_name, bucket_name, file_name)

# Function to load FAISS index from S3
def load_faiss_index_from_s3(bucket_name, file_name):
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    s3_client.download_file(bucket_name, file_name, file_name)
    return faiss.read_index(file_name)

# Function to use LLM for generating a response
def generate_response(prompt):
    response = bedrock_client.invoke_model(
        ModelId='anthropic.sonnet',
        ContentType='application/json',
        Body={"text": prompt}
    )
    return response['Body']['generated_text']

# Memory buffer to maintain chat history
chat_history = deque(maxlen=5)

# Function to evaluate the model
def evaluate_responses(ground_truth, generated_response):
    sim_score = cosine_similarity([ground_truth], [generated_response])
    return sim_score[0][0]

# Guardrails to restrict responses
def apply_guardrails(response):
    allowed_topics = ['GenAI Hub']
    if not any(topic in response for topic in allowed_topics):
        return "Hmm, I don't know"
    return response

# Main function to run the RAG pipeline
def main():
    # Step 1: Extract text from PDF
    pdf_path = 'your_file.pdf'
    extracted_text = extract_text_from_pdf(pdf_path)

    # Step 2: Split text using Recursive Character Splitter
    text_chunks = recursive_character_splitter(extracted_text)

    # Step 3: Get embeddings for the text chunks
    embeddings = np.array([get_embeddings(chunk) for chunk in text_chunks])

    # Step 4: Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = create_faiss_index(embeddings, dimension)

    # Step 5: Store FAISS index in S3
    bucket_name = 'your_bucket_name'
    index_file_name = 'faiss_index.bin'
    store_faiss_index_in_s3(index, bucket_name, index_file_name)

    # Step 6: Load FAISS index from S3
    index = load_faiss_index_from_s3(bucket_name, index_file_name)

    # Step 7: User input query and convert to embedding
    user_query = input("Enter your query: ")
    query_embedding = get_embeddings(user_query)

    # Step 8: Retrieve relevant documents from FAISS
    distances, indices = retrieve_from_faiss(query_embedding, index)

    # Step 9: Prepare prompt with retrieved information and chat history
    prompt = " ".join([text_chunks[idx] for idx in indices[0]])
    chat_history.append(user_query)
    prompt_with_history = "\n".join(chat_history) + "\n" + prompt

    # Step 10: Generate response using LLM
    generated_response = generate_response(prompt_with_history)

    # Step 11: Apply guardrails to the response
    safe_response = apply_guardrails(generated_response)

    # Step 12: Evaluate the response
    evaluation_score = evaluate_responses(prompt, safe_response)
    print(f"Evaluation Score (Cosine Similarity): {evaluation_score}")

    # Display the generated response
    print(f"Generated Response: {safe_response}")

if __name__ == "__main__":
    main()
