import boto3
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import json
import os
import requests
from bs4 import BeautifulSoup
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# AWS Credentials (hardcoded for simplicity; consider using environment variables in production)
# AWS_ACCESS_KEY = 'ASIA47CR2CBK6YW6KAQH'
# AWS_SECRET_KEY = 'nppi9nO/zvAy+1DQbZ1/mSrk7m8+QOwJk621MTL0'
# AWS_REGION = 'ap-southeast-2'

AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")

access_key_id = 'XXXX'
secret_access_key = 'YYYY'
session_token = 'ZZZZ'

session = boto3.Session(
                       aws_access_key_id=access_key_id,
                       aws_secret_access_key=secret_access_key,
                       aws_session_token=session_token)

sts_client = session.client('sts')

response = sts_client.assume_role(
       RoleArn="arn:aws:iam::1234567890:role/AdminUserRole",
       RoleSessionName="AdminUserRoleSession"
   )

credentials = response['Credentials']

bedrock_client = boto3.client(service_name='bedrock-runtime', 
                             aws_access_key_id = credentials['AccessKeyId'],
                             aws_secret_access_key = credentials['SecretAccessKey'],
                             aws_session_token = credentials['SessionToken'],
                             region_name=AWS_REGION)

s3_client = boto3.client(service_name='s3', 
                           aws_access_key_id = credentials['AccessKeyId'],
                           aws_secret_access_key = credentials['SecretAccessKey'],
                           aws_session_token = credentials['SessionToken'],
                           region_name=AWS_REGION)

# Initialize Boto3 Bedrock Client
# bedrock_client = boto3.client(
#    'bedrock-runtime',
#    region_name=AWS_REGION,
#    aws_access_key_id=AWS_ACCESS_KEY,
#    aws_secret_access_key=AWS_SECRET_KEY
# )

# Function to extract text and hyperlinks from PDF
def extract_text_and_hyperlinks(pdf_path):
  reader = PdfReader(pdf_path)
  text = ''
  hyperlinks = []

  for page in reader.pages:
      text += page.extract_text()

      # Extract hyperlinks from annotations (handling IndirectObject)
      if "/Annots" in page:
          annotations = page["/Annots"]
          for annot in annotations:
              annot_obj = annot.get_object()
              if "/A" in annot_obj and "/URI" in annot_obj["/A"]:
                  uri = annot_obj["/A"]["/URI"]
                  hyperlinks.append(uri)

  return text, hyperlinks

# Fetch the content from hyperlinks (HTML pages)
def fetch_hyperlink_content(url):
  try:
      response = requests.get(url, verify=False)
      soup = BeautifulSoup(response.text, 'html.parser')
      return soup.get_text()
  except Exception as e:
   #   print(f"Error fetching {url}")
      return ""

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

   max_length = 2048

   if len(text) > max_length:
       # print(f"Text length exceeds {max_length} characters. Truncating to fit.")
       text = text[:max_length]

   input_data = json.dumps({
      'texts': [text], # Using 'texts' as a list
      'input_type': 'search_document'
  }).encode('utf-8')

   response = bedrock_client.invoke_model(
       modelId='cohere.embed-english-v3',
       contentType='application/json',
       body=input_data
   )

   # print("###################")
   # print("Hari response: ")
   # print(response)
   # print("###################")
   response_body = json.loads(response['body'].read().decode('utf-8')) # Read and parse the response body
   # print("###################")
   # print("Hari response: ")
   # print(response_body)
   # print("###################")
   embedding = np.array(response_body['embeddings'][0])
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
   # s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
   s3_client.upload_file(file_name, bucket_name, file_name)

# Function to load FAISS index from S3
def load_faiss_index_from_s3(bucket_name, file_name):
   # s3_client = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
   s3_client.download_file(bucket_name, file_name, file_name)
   return faiss.read_index(file_name)

# Function to use LLM for generating a response
def generate_response(prompt):

   #formatted_prompt = f'Human: {context}\n\n{prompt}\nAssistant:'
   input_data = json.dumps({
      "anthropic_version": "bedrock-2023-05-31",
   #   'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
      'messages' : [
               # {'role':'system', 'content': 'You are a helpful assistant'},
               {'role':'user', 'content':prompt}
           ],
   #   'prompt': formatted_prompt,x
      "max_tokens": 1024, 
   #   'max_tokens_to_sample': 100,
      'temperature': 0.1,
      'top_p': 0.9,
   #   'input_type': 'text'
  }).encode('utf-8')

   response = bedrock_client.invoke_model(
       modelId='anthropic.claude-3-sonnet-20240229-v1:0',
       contentType='application/json',
       body=input_data
   )

   # print("####################")
   # print("Hari response generate_response: ")
   # print(response)
   # print("#####################")

   # print("###########################")
   # print("Hari test", response_body)
   # print("###########################")

   response_body = json.loads(response['body'].read().decode('utf-8')) # Read and parse the response body
   return response_body['content'][0]['text']

# Memory buffer to maintain chat history
chat_history = deque(maxlen=5)

# Function to evaluate the model
def evaluate_responses(ground_truth, generated_response):

   # print("ground_truth_type", type(ground_truth))
   # print("generated_response_type", type(generated_response))
   # if isinstance(ground_truth, str) or isinstance(generate_response, str):
   #    raise ValueError("Inputs must be numeric embedding, not strings.")

   sim_score = cosine_similarity([ground_truth], [generated_response])
   return sim_score[0][0]

# Guardrails to restrict responses
def apply_guardrails(response):
   allowed_topics = ['GenAI']
   if not any(topic in response for topic in allowed_topics):
       return "I can only answer question related to GenAI."
   return response

# Main function to run the RAG pipeline
def retrieve_knowledge_hyper(user_query):
   # # Step 1: Extract text from PDF
   pdf_path = 'Gen_AI_Playbook_20240704.pdf'
   #extracted_text = extract_text_from_pdf(pdf_path)

   extracted_text, hyperlinks = extract_text_and_hyperlinks(pdf_path)
   # print("hyperlinks: ", hyperlinks)

   # Step 2: Fetch content from hyperlinks and add to extracted text
   hyperlink_content = ""
   # for link in hyperlinks:
   #    # print("hyperlinks: ", link)
   #    hyperlink_content += fetch_hyperlink_content(link)

   # for index, link in enumerate(hyperlinks, start=1):
   #    # print(f"{index} ==> {link}")
   #    hyperlink_content += fetch_hyperlink_content(link)
   # print("hyperlink_content: ", hyperlink_content) 
   # print("type hyperlink_content: ", type(hyperlink_content)) 
   # print("Len of hyperlink_content: ", len(hyperlink_content))

   normalized_text = ""
   # normalized_text = ' '.join(hyperlink_content.split())
   # normalized_text = normalized_text.replace("GitHub", "")
   # normalized_text = normalized_text.replace("FORBIDDEN", "")
   # hyperlink_content = re.sub(r'(?i)\b(GitHub|FORBIDDEN)\b', '', hyperlink_content)

   # print("type normalized_text_hyperlink_content: ", type(normalized_text)) 
   # print("Len of normalized_text_hyperlink_content: ", len(normalized_text))
   # print("=========")
   # print("normalized_text_hyperlink_content: ", normalized_text) 

  # Combine PDF text and hyperlink content
   combined_text = extracted_text + "\n" + normalized_text

   # # Step 2: Split text using Recursive Character Splitter
   text_chunks = recursive_character_splitter(combined_text)

   # # Step 3: Get embeddings for the text chunks
   # embeddings = np.array([get_embeddings(chunk) for chunk in text_chunks])

   # # Step 4: Create and populate FAISS index
   # dimension = embeddings.shape[1]
   # index = create_faiss_index(embeddings, dimension)

   # # Step 5: Store FAISS index in S3
   bucket_name = 'cmp-demo-llm'
   index_file_name = 'faiss_full_4_index.bin'
   # store_faiss_index_in_s3(index, bucket_name, index_file_name)

   # Step 6: Load FAISS index from S3
   index = load_faiss_index_from_s3(bucket_name, index_file_name)

   # Step 7: User input query and convert to embedding
   #user_query = input("Enter your query: ")
   # user_query = "What is the main genai topic discussed in the document?"
   query_embedding = get_embeddings(user_query)

   # Step 8: Retrieve relevant documents from FAISS
   distances, indices = retrieve_from_faiss(query_embedding, index)

   # Step 9: Prepare prompt with retrieved information and chat history
   prompt = " ".join([text_chunks[idx] for idx in indices[0]])
   chat_history.append(user_query)
   prompt_with_history = "\n".join(chat_history) + "\n" + prompt

   # Step 10: Generate response using LLM
   # generated_response = generate_response(prompt_with_history)


   input_data = json.dumps({
      "anthropic_version": "bedrock-2023-05-31",
   #   'modelId': 'anthropic.claude-3-sonnet-20240229-v1:0',
      'messages' : [
               # {'role':'system', 'content': 'You are a helpful assistant'},
               {'role':'user', 'content':prompt_with_history}
           ],
   #   'prompt': formatted_prompt,x
      "max_tokens": 1024, 
   #   'max_tokens_to_sample': 100,
      'temperature': 0.1,
      'top_p': 0.9,
   #   'input_type': 'text'
  }).encode('utf-8')

   response = bedrock_client.converse_stream(
       modelId='anthropic.claude-3-sonnet-20240229-v1:0',
       # contentType='application/json',
       messages=[
               # {'role':'system', 'content': 'You are a helpful assistant'},
               {'role':'user', 'content':[{'text': prompt_with_history}]}
           ]
   )
   print("Hari stream bedrock hyperlink frontend")
   print("response")
   stream = response.get('stream')
   if stream:
       for event in stream:
           if 'messageStart' in event:
               print(f"\nRole: {event['messageStart']['role']}")

           if 'contentBlockDelta' in event:
               print(event['contentBlockDelta']['delta']['text'], end="")
               yield event['contentBlockDelta']['delta']['text']

           if 'messageStop' in event:
               print(f"\nStop reason: {event['messageStop']['stopReason']}")

           if 'metadata' in event:
               metadata = event['metadata']
               if 'usage' in metadata:
                   print("\nToken usage")
                   print(f"Input tokens: {metadata['usage']['inputTokens']}")
                   print(
                       f":Output tokens: {metadata['usage']['outputTokens']}")
                   print(f":Total tokens: {metadata['usage']['totalTokens']}")
               if 'metrics' in event['metadata']:
                   print(
                       f"Latency: {metadata['metrics']['latencyMs']} milliseconds")
   # Step 11: Apply guardrails to the response
   # safe_response = apply_guardrails(generated_response)

   # prompt_embedding = get_embeddings(prompt_with_history)
   # response_embedding = get_embeddings(safe_response)

   # Step 12: Evaluate the response
   # evaluation_score = evaluate_responses(prompt_embedding, response_embedding)
   # print(f"Evaluation Score (Cosine Similarity): {evaluation_score}")

   # Display the generated response
   # print("Hackathon Response: ")
   # print(f"{safe_response}")

if __name__ == "__main__":
   user_query = "What is the main GenAI topic discussed in the document?"
   retrieve_knowledge_hyper(user_query)
