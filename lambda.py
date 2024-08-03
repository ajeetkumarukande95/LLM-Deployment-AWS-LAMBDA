import json
import boto3

# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name='us-east-1')

# The endpoint name must be unique within 
# an AWS Region in your AWS account. 
endpoint_name='huggingface-pytorch-tgi-inference-2024-08-03-14-56-54-652'

def lambda_handler(event, context):
    query_params = event['queryStringParameters']
    query = query_params.get('query')
    payload = {
    "inputs" : query,
    "parameters" : {
        "do_sample" : True,
        "top_p" : 0.9,
        "temperature" : 0.3,
        "top_k" : 50,
        "max_new_tokens" : 256,
        "repition_penalty" : 1.03}}
    response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name, 
    Body=json.dumps(payload),
    ContentType = "application/json"
    )
    predictions = json.loads(response['Body'].read().decode('utf-8'))
    final_result = predictions[0]['generated_text']
    return {
        'statusCode': 200,
        'body': json.dumps(final_result)
    }
