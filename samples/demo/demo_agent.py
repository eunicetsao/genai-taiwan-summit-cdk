import json

import langchain_visualizer
from langchain import SQLDatabase
from langchain import SagemakerEndpoint
from langchain.agents import create_sql_agent, AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
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
    endpoint_name='huggingface-pytorch-tgi-inference-2023-07-18-08-38-00-466',
    region_name='us-east-1',
    model_kwargs={"temperature": 0.01, "max_new_tokens": 200},
    content_handler=content_handler
)

ATHENA_BUCKET = 'genai-text-to-sql-workshop-data7e2128ca-fxi6ydpzyhrd'
ATHENA_DATABASE = 'genai-text-to-sql-workshop'
ATHENA_REGION = 'us-east-1'
conn_str = f"awsathena+rest://:@athena.{ATHENA_REGION}.amazonaws.com:443/{ATHENA_DATABASE}?s3_staging_dir=s3://{ATHENA_BUCKET}/Unsaved/"
database_engine = create_engine(conn_str)
data_base = SQLDatabase(database_engine, sample_rows_in_table_info=0)

question = "What is total sale amount of Fruits"

toolkit = SQLDatabaseToolkit(db=data_base, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)


async def search_agent_demo():
    return agent_executor(
        question
    )


langchain_visualizer.visualize(search_agent_demo)
