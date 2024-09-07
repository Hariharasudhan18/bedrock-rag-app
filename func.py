import streamlit as st
import boto3
from botocore.exceptions import ClientError
import os

AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
KB_ID = 'ASASPO9ZA8' # LLUECHU1GA Bedrock knowledge base ID

access_key_id = 'XXXX'
secret_access_key = 'YYYY'
session_token = 'ZZZZ'

session = boto3.Session(
                       aws_access_key_id=access_key_id,
                       aws_secret_access_key=secret_access_key,
                       aws_session_token=session_token)

sts_client = session.client('sts')

response = sts_client.assume_role(
       RoleArn="arn:aws:iam::12345678:role/AdminUserRole",
       RoleSessionName="AdminUserRoleSession"
   )

credentials = response['Credentials']

bedrock_client = boto3.client(service_name='bedrock-runtime', 
                             aws_access_key_id = credentials['AccessKeyId'],
                             aws_secret_access_key = credentials['SecretAccessKey'],
                             aws_session_token = credentials['SessionToken'],
                             region_name=AWS_REGION)
bedrock_agent_client = boto3.client(service_name='bedrock-agent-runtime', 
                                   aws_access_key_id = credentials['AccessKeyId'],
                                   aws_secret_access_key = credentials['SecretAccessKey'],
                                   aws_session_token = credentials['SessionToken'],
                                   region_name=AWS_REGION)

# bedrock_client = boto3.client(service_name='bedrock-runtime', 
#                              aws_access_key_id = access_key_id,
#                              aws_secret_access_key = secret_access_key,
#                              aws_session_token = session_token,
#                              region_name=AWS_REGION)
# bedrock_agent_client = boto3.client(service_name='bedrock-agent-runtime', 
#                                    aws_access_key_id = access_key_id,
#                                    aws_secret_access_key = secret_access_key,
#                                    aws_session_token = session_token,
#                                    region_name=AWS_REGION)

def update_session_state(key: str, value: str):
   """
   update session
   """
   st.session_state[key] = value


def stream_bedrock_response(model_id,
                           messages,
                           system_prompts,
                           inference_config,
                           addition_model_fields={}):
   """
   Sends messages to a model and streams the response.
   Args:
       model_id (str): The model ID to use.
       messages (JSON) : The messages to send.
       system_prompts (JSON) : The system prompts to send.
       inference_config (JSON) : The inference configuration to use.
       additional_model_fields (JSON) : Additional model fields to use.
   Returns:
       Nothing.
   """
   print("Streaming messages with model %s", model_id)

   # Invoke Bedrock Converse API
   try:
       response = bedrock_client.converse_stream(
           modelId=model_id,
           messages=messages,
           system=system_prompts,
           inferenceConfig=inference_config,
           additionalModelRequestFields=addition_model_fields,
       )

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
   except ClientError as err:
           message = err.response['Error']['Message']
           print("A client error occurred: " +
               format(message))
   else:
       print(
           f"Finished streaming messages with model {model_id}.")


def bedrock_response(model_id,
                    messages,
                    system_prompts,
                    inference_config):
   """
   Sends messages to a model and return the response.
   Args:
       model_id (str): The model ID to use.
       messages (JSON) : The messages to send.
       system_prompts (JSON) : The system prompts to send.
       inference_config (JSON) : The inference configuration to use.
       additional_model_fields (JSON) : Additional model fields to use.
   Returns:
       Full response.
   """
   print("Return messages with model %s", model_id)

   # Invoke Bedrock Converse API
   try:
       response = bedrock_client.converse(
           modelId=model_id,
           messages=messages,
           system=system_prompts,
           inferenceConfig=inference_config
       )
       return response
   except ClientError as err:
           message = err.response['Error']['Message']
           print("A client error occurred: " +
               format(message))


def retrieve_knowledge(query):
   """
   Function to retrieve results from the knowledge base
   """
   response = bedrock_agent_client.retrieve(
       knowledgeBaseId=KB_ID,
       retrievalConfiguration={
           'vectorSearchConfiguration': {
               'numberOfResults': 4
           }
       },
       retrievalQuery={
           'text': query
       }
   )
   return response['retrievalResults']
