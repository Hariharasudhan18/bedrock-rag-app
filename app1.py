import boto3
import PyPDF2
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# # Hardcoded AWS credentials (not recommended for production)
# aws_access_key_id = 'YOUR_ACCESS_KEY'
# aws_secret_access_key = 'YOUR_SECRET_KEY'
# aws_region_name = 'us-west-2' # Change to your preferred region

# # Initialize the Boto3 client for AWS Bedrock
# bedrock_client = boto3.client(
#    'bedrock',
#    region_name=aws_region_name,
#    aws_access_key_id=aws_access_key_id,
#    aws_secret_access_key=aws_secret_access_key
# )

AWS_REGION = "ap-southeast-2"

access_key_id = 'XXXX'
secret_access_key = 'YYYY'
session_token = 'ZZZZ'

session = boto3.Session(
                        aws_access_key_id=access_key_id,
                        aws_secret_access_key=secret_access_key,
                        aws_session_token=session_token)

sts_client = session.client('sts')

response = sts_client.assume_role(
        RoleArn="arn:aws:iam::891377291349:role/AdminUserRole",
        RoleSessionName="AdminUserRoleSession"
    )

credentials = response['Credentials']

bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              aws_access_key_id = credentials['AccessKeyId'],
                              aws_secret_access_key = credentials['SecretAccessKey'],
                              aws_session_token = credentials['SessionToken'],
                              region_name=AWS_REGION)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
   with open(pdf_file_path, 'rb') as file:
       reader = PyPDF2.PdfReader(file)
       text = ''
       for page in reader.pages:
           text += page.extract_text()
   return text

# Function to embed text using Cohere Embed English v3 via Bedrock API
def embed_text(text):
   # Correct the input format for the embedding model
   input_data = json.dumps({
       'texts': [text], # Using 'texts' as a list
       'input_type': 'search_document'
   }).encode('utf-8')

   response = bedrock_client.invoke_model(
       modelId='cohere.embed-english-v3', # Change to the appropriate model ID for Cohere Embed
       body=input_data,
       contentType='application/json',
       accept='application/json'
   )
   response_body = json.loads(response['body'].read().decode('utf-8')) # Read and parse the response body
   embeddings = response_body['embeddings'][0] # Assuming the response contains an 'embeddings' field
   return np.array(embeddings)

# Function to generate response using Anthropic's Sonnet via Bedrock API
def generate_response(prompt, context):
   # Correct the input format for the LLM model
#    print("Hari test", prompt)
#    input_data = json.dumps({
#                         'prompt': f"{context}\n\n{prompt}",
#                         'max_tokens_to_sample': 100,
#                         'temperature': 0.1,
#                         'top_p': 0.9,
#                         #    'input_type': 'text'
#                     }).encode('utf-8')
#    print("Hari prompt", input_data)

   formatted_prompt = f'Human: {context}\n\n{prompt}\nAssistant:'
   
   input_data = json.dumps({
       "anthropic_version": "bedrock-2023-05-31",
    #    'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
       'messages' : [
                # {'role':'system', 'content': 'You are a helpful assistant'},
                {'role':'user', 'content':formatted_prompt}
            ],
    #    'prompt': formatted_prompt,x
       "max_tokens": 1024, 
    #    'max_tokens_to_sample': 100,
       'temperature': 0.1,
       'top_p': 0.9,
    #    'input_type': 'text'
   }).encode('utf-8')

   response = bedrock_client.invoke_model(
       modelId='anthropic.claude-3-sonnet-20240229-v1:0', # Change to the appropriate model ID for Anthropic Sonnet
       body=input_data,
       contentType='application/json',
       accept='application/json'
   )
   response_body = json.loads(response['body'].read().decode('utf-8')) # Read and parse the response body
   
   print("###########################")
   print("Hari test", response_body)
   print("###########################")
   generated_text = response_body['content'][0]['text'] # Assuming the response contains an 'output' field
   return generated_text

# Function to check for NaN values and reshape embeddings
def preprocess_embeddings(embedding):
   # Convert to numpy array if not already
   embedding = np.array(embedding)

   # Check for NaN values and replace them with 0 (or handle as needed)
   if np.any(np.isnan(embedding)):
       embedding = np.nan_to_num(embedding)

   # Reshape to 2D array (1, -1) for a single sample
   return embedding.reshape(1, -1)

# Main function to execute RAG process
def main():
   # Path to the PDF file
   pdf_file_path = 'Gen_AI_Playbook_20240704.pdf'

   # Extract text from PDF
   document_text = extract_text_from_pdf(pdf_file_path)

   # Split text into chunks (simplified example)
   text_chunks = [document_text[i:i+500] for i in range(0, len(document_text), 500)]

   # Embed text chunks
   embeddings = [preprocess_embeddings(embed_text(chunk)) for chunk in text_chunks]
   
   # User query
   user_query = "What is the main topic discussed in the document?"

   # Embed user query
   query_embedding = preprocess_embeddings(embed_text(user_query))

   # Calculate cosine similarity between query and document chunks
   similarities = [cosine_similarity(query_embedding, embedding)[0][0] for embedding in embeddings]

   # Find the most relevant chunk
   most_relevant_chunk = text_chunks[np.argmax(similarities)]

   # Generate response using the most relevant chunk as context
   generated_response = generate_response(user_query, most_relevant_chunk)

   print("Generated Response:", generated_response)

if __name__ == "__main__":
   main()
