# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shows how to generate text embeddings using the Cohere Embed English model.
"""
import json
import logging
import boto3


from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

bedrock = boto3.client(service_name='bedrock-runtime', 
                             aws_access_key_id = credentials['AccessKeyId'],
                             aws_secret_access_key = credentials['SecretAccessKey'],
                             aws_session_token = credentials['SessionToken'],
                             region_name=AWS_REGION)

def generate_text_embeddings(model_id, body):
   """
   Generate text embedding by using the Cohere Embed model.
   Args:
       model_id (str): The model ID to use.
       body (str) : The reqest body to use.
   Returns:
       dict: The response from the model.
   """

   logger.info(
       "Generating text emdeddings with the Cohere Embed model %s", model_id)

   accept = '*/*'
   content_type = 'application/json'

   # bedrock = boto3.client(service_name='bedrock-runtime')

   response = bedrock.invoke_model(
       body=body,
       modelId=model_id,
       accept=accept,
       contentType=content_type
   )

   logger.info("Successfully generated text with Cohere model %s", model_id)

   return response


def main():
   """
   Entrypoint for Cohere Embed example.
   """

   logging.basicConfig(level=logging.INFO,
                       format="%(levelname)s: %(message)s")

   model_id = 'cohere.embed-english-v3'
   text1 = "hello world"
   text2 = "this is a test"
   input_type = "search_document"
   embedding_types = ["int8", "float"]

   try:

       body = json.dumps({
           "texts": [
               text1,
               text2],
           "input_type": input_type,
           "embedding_types": embedding_types}
       )
       response = generate_text_embeddings(model_id=model_id,
                                           body=body)

       response_body = json.loads(response.get('body').read())

       print(f"ID: {response_body.get('id')}")
       print(f"Response type: {response_body.get('response_type')}")

       print("Embeddings")
       for i, embedding in enumerate(response_body.get('embeddings')):
           print(f"\tEmbedding {i}")
           print(*embedding)

       print("Texts")
       for i, text in enumerate(response_body.get('texts')):
           print(f"\tText {i}: {text}")

   except ClientError as err:
       message = err.response["Error"]["Message"]
       logger.error("A client error occurred: %s", message)
       print("A client error occured: " +
             format(message))
   else:
       print(
           f"Finished generating text embeddings with Cohere model {model_id}.")


if __name__ == "__main__":
   main()
