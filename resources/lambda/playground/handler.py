import json
import os

from langchain import SQLDatabaseChain, SQLDatabase
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from sqlalchemy import create_engine


class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        input_str = json.dumps({"inputs": prompt, "parameters": model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]


content_handler = ContentHandler()
llm = SagemakerEndpoint(
    endpoint_name='huggingface-pytorch-tgi-inference-2023-07-16-05-23-44-657',
    region_name='us-east-1',
    model_kwargs={"temperature": 0.01, "max_new_tokens": 200},
    content_handler=content_handler
)


def get_athena_database():
    ATHENA_BUCKET = os.getenv('ATHENA_BUCKET')
    ATHENA_DATABASE = os.getenv('ATHENA_DATABASE')
    ATHENA_REGION = os.getenv('ATHENA_REGION')

    conn_str = f"awsathena+rest://:@athena.{ATHENA_REGION}.amazonaws.com:443/{ATHENA_DATABASE}?s3_staging_dir=s3://{ATHENA_BUCKET}/Unsaved/"
    database_engine = create_engine(conn_str)
    data_base = SQLDatabase(database_engine)

    return data_base


def lambda_handler(event, context):
    question = event['question']

    data_base = get_athena_database()

    db_chain = SQLDatabaseChain.from_llm(llm, data_base, verbose=True)

    result = db_chain(question)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps({
            'sql': result
        })
    }
