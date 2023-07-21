import json
import os
from typing import Dict, Any, Optional, List

from langchain import SQLDatabaseChain, SQLDatabase, PromptTemplate, LLMChain
from langchain import SagemakerEndpoint
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.sql_database.base import INTERMEDIATE_STEPS_KEY
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from sqlalchemy import create_engine


class SQLDatabaseChainWithInsight(SQLDatabaseChain):
    return_intermediate_steps: bool = True

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        table_info = self.database.get_table_info(table_names=table_names_to_use)
        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            "stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)  # input: sql generation
            sql_cmd = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()

            _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
            intermediate_steps.append(
                sql_cmd
            )  # output: sql generation (no checker)
            intermediate_steps.append({"sql_cmd": sql_cmd})  # input: sql exec
            result = self.database.run(sql_cmd)
            intermediate_steps.append(str(result))  # output: sql exec

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            _run_manager.on_text("\nAnswer:", verbose=self.verbose)
            input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
            llm_inputs["input"] = input_text
            intermediate_steps.append(llm_inputs)  # input: final answer
            sql_data = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()
            intermediate_steps.append(sql_data)  # output: sql data
            _run_manager.on_text(sql_data, color="green", verbose=self.verbose)

            GET_INSIGHT = """
                    You are a senior data analytics.

                    Your task is to analyze the given company data in JSON format and provide insights or explanations for any trends or patterns observed. The data pertains to the question: 
                    {question}.
                    Your response should be clear and concise, no more than 200 words.

                    Response data: {data}

                    Please note that if the data is empty or null, you should simply state "no insight." 

                    My Insight:
                    """
            get_insight_prompt = PromptTemplate(
                template=GET_INSIGHT, input_variables=["question", "data"]
            )
            get_insight_chain = LLMChain(
                llm=self.llm_chain.llm, prompt=get_insight_prompt
            )

            get_insight_inputs = {
                "question": inputs[self.input_key],
                "data": result,
            }

            final_result: str = get_insight_chain.predict(
                callbacks=_run_manager.get_child(), **get_insight_inputs
            ).strip()

            _run_manager.on_text(
                final_result, color="blue", verbose=self.verbose
            )

            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc


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

    db_chain = SQLDatabaseChainWithInsight.from_llm(llm, data_base, verbose=True)

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
