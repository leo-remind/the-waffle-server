import io
import json
import logging
import os
import re
from asyncio import sleep
from typing import List

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from fastapi import WebSocket
from langchain_openai import ChatOpenAI
from pinecone import Pinecone
from rich.pretty import pretty_repr
from supabase import Client, create_client
from langchain_anthropic import ChatAnthropic

from constants import RETRIEVAL_THRESHOLD
from prompts import (
    GRAPH_GENERATION_PROMPT,
    NORMAL_RESPONSE_PROMPT,
    QUERY_AUGMENTATION_PROMPT,
    SQL_GENERATION_PROMPT,
)

load_dotenv()
logger = logging.getLogger(__file__)

MODEL_NAME_RE = re.compile(
    r"SELECT.*?FROM\s*\"?(.{20})\"?.*", re.IGNORECASE | re.MULTILINE | re.DOTALL
)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

pg_user = os.getenv("PG_USER")
pg_password = os.getenv("PG_PASSWORD")
pg_host = os.getenv("PG_HOST")
pg_port = os.getenv("PG_PORT")
pg_dbname = os.getenv("PG_DBNAME")
pg_conn = psycopg2.connect(
    host=pg_host,
    database=pg_dbname,
    user=pg_user,
    password=pg_password,
    port=pg_port,
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
chatgpt_4o = ChatOpenAI(model="gpt-4o")
chatgpt_o3_mini = ChatOpenAI(model="o3-mini")
claude_3_7 = ChatAnthropic(
    model="claude-3-7-sonnet-20250219",  # Use the Claude 3.7 Sonnet model
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens_to_sample=8092
)


def stream_fmt(a):
    return f"data: {a}\n\n"


# stream_fmt = lambda a: f"data: {a}\n\n"


async def query_rag(
    websocket: WebSocket,
    og_query: str,
    verbose: bool = False,
    graph: bool = False,
    pdf_name: str | None = None,
):
    await websocket.send_text(
        json.dumps({"isStreaming": True, "message": "Finding most relevant tables\n\n"})
    )
    query = query_augmentation(og_query)
    tables = find_k_relevant_tables(query, pdf_name)

    if not tables:
        await websocket.send_text(
            json.dumps(
                {
                    "isStreaming": False,
                    "message": "no relevant tables found",
                    "tables": [],
                }
            )
        )

    await sleep(0.1)
    schemas = get_table_schemas(tables)
    schema_str = "\n\n".join(
        [
            f"-- table topic: {table['text']}\n{schema}"
            for table, schema in zip(tables, schemas)
        ]
    )
    sql_reasoning, sql_query = generate_sql_query(query, schema_str)

    if sql_query:
        model_name_match = MODEL_NAME_RE.match(sql_query)
    else:
        model_name_match = None
    if model_name_match is not None:
        primary_table = model_name_match.groups()[0]
    else:
        primary_table = ""
    ############################### TABLES & SQL Query ####################################################
    await websocket.send_text(
        json.dumps(
            {
                "isStreaming": True,
                "message": "Querying SQL Database with query",
                "sqlQuery": sql_query,
                "sqlReasoning": sql_reasoning,
                "primaryTable": primary_table,
                "tables": [
                    get_table_as_json(table["supabase_table_name"]) for table in tables
                ],
            }
        )
    )
    ##############################################################################################
    await sleep(0.1)
    results = execute_sql_query(sql_query)

    await websocket.send_text(
        json.dumps(
            {"isStreaming": True, "message": "Values collated, generating response..."}
        )
    )
    await sleep(0.1)
    response = generate_response(
        og_query, results, sql_reasoning, schema_str, verbose=verbose, graph=graph
    )

    ############################### ChartJS Graph ####################################################
    graph_code = None
    if graph:
        await websocket.send_text(
            json.dumps({"isStreaming": True, "message": "Creating graph for your use-case"})
        )

        graph_code  = graph_data(primary_table, query, sql_query, schema_str)

    await sleep(0.1)
    ##################################################################################################

    await websocket.send_text(
        json.dumps(
            {
                "isStreaming": False,
                "message": response,
                "graph_code" : graph_code,
            }
        )
    )
    await sleep(0.1)
    logger.info(response)


def get_table_as_json(table_name: str) -> str:
    response = supabase.table(table_name).select("*").csv().execute()
    df = pd.read_csv(io.StringIO(response.data))
    data = df.to_json(index=False)

    resp = (
        supabase.table("METADATA")
        .select("supabase_table_name", "table_heading, pdf_url, page_number")
        .eq("supabase_table_name", table_name)
        .execute()
    )

    meta = resp.data[0]

    return json.dumps(
        {
            "meta": meta,
            "data": json.loads(data),
        }
    )


def find_k_relevant_tables(
    query: str, pdf_name: str | None = None, top_k: int = 3
) -> List[str]:
    index = pc.Index("the-waffle")
    query = {
        "inputs": {"text": query},
        "top_k": top_k * 4,
    }

    if pdf_name is not None:
        logger.info(f"FILTER pinecone on pdf_name: {pdf_name}")
        query["filter"] = {"pdf_name": pdf_name}

    results = index.search_records(
        namespace="",
        query=query,
        fields=["text", "supabase_table_name"],
        rerank={"model": "bge-reranker-v2-m3", "top_n": top_k, "rank_fields": ["text"]},
    )

    hits = results["result"]["hits"]
    hits = filter(lambda x: x["_score"] > RETRIEVAL_THRESHOLD, hits)

    top_k_tables = [hit["fields"] for hit in hits]
    logger.info(pretty_repr(top_k_tables))

    return top_k_tables


def extract_first_code_block(text):
    # Try to find triple quotes first
    triple_quote_start = text.find('"""')
    if triple_quote_start != -1:
        # Find the end of the triple quotes block
        triple_quote_end = text.find('"""', triple_quote_start + 3)
        if triple_quote_end != -1:
            # Extract content between triple quotes
            return text[triple_quote_start + 3 : triple_quote_end]

    # If no triple quotes found, try triple backticks
    backtick_start = text.find("```")
    if backtick_start != -1:
        # Find the end of the triple backticks block
        backtick_end = text.find("```", backtick_start + 3)
        if backtick_end != -1:
            # Extract content between backticks
            content = text[backtick_start + 3 : backtick_end]

            # Remove language hint if present (text before the first newline)
            first_newline = content.find("\n")
            if first_newline != -1:
                # Check if there's text before the newline (indicating a language hint)
                if first_newline > 0:
                    # Skip the language hint and the newline
                    return content[first_newline + 1 :]
                else:
                    # Just a newline with no language hint
                    return content[1:]
            else:
                # No newline found, return the content as is
                return content

    # No code blocks found
    return None


def query_augmentation(query: str, llm: ChatOpenAI = chatgpt_4o) -> str:
    chain = QUERY_AUGMENTATION_PROMPT | llm

    response = chain.invoke(query)
    logger.info(f"augmented query: {response.content}")
    aug_query = extract_first_code_block(response.content)
    return aug_query


def get_table_schemas(tables: List):
    global supabase
    schemas = []
    for table in tables:
        response = (
            supabase.table("METADATA")
            .select("schema")
            .eq("supabase_table_name", table["supabase_table_name"])
            .execute()
        )
        schemas.append(response.data[0]["schema"])
    logger.info(schemas)
    return schemas


def generate_sql_query(query: str, schema_str: str, llm: ChatOpenAI = chatgpt_o3_mini):
    chain = SQL_GENERATION_PROMPT | llm
    logger.info(f"query: {query}")

    response = chain.invoke({"query": query, "schema": schema_str})
    logger.info(f"generated_response: {response.content}")
    sql_query = extract_first_code_block(response.content)
    logger.info(f"generated sql query:\n {sql_query}")

    return response.content, sql_query


def execute_sql_query(sql_query: str):
    global pg_conn

    sql_query = sql_query.strip("`").removeprefix("sql")
    cursor = pg_conn.cursor()
    cursor.execute(sql_query)

    # cursor.execute("ROLLBACK")
    results = cursor.fetchall()

    cursor.close()
    logger.info(f"Executed query, results: {pretty_repr(results)}")

    return results


def generate_response(
    query: str,
    results: any,
    sql_reasoning: str,
    sql_schema: str,
    llm: ChatOpenAI = chatgpt_4o,
    verbose: bool = False,
    graph: bool = False,
):
    prompt = NORMAL_RESPONSE_PROMPT  # VERBOSE_RESPONSE_PROMPT if verbose else NORMAL_RESPONSE_PROMPT
    chain = prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "result": str(results),
            "sql_reasoning": sql_reasoning,
            "schema" : sql_schema,
        }
    )

    return response.content


import tempfile
import os
from typing import Dict, Any, Optional

def graph_data(
    table: str,
    query: str,
    sql_query: str,
    schema: str,
    llm: ChatOpenAI = claude_3_7,
):
    data = get_table_data_as_csv(table) 
    chain = GRAPH_GENERATION_PROMPT | llm

    response = chain.invoke(
        {"query": query, "data": data},
    )

    logger.info(response.content)
    svg = extract_first_code_block(response.content)
    print(svg)
    return svg
    

def get_table_data_as_csv(table_name):
    """
    Fetch data from a Supabase table and save it as CSV
    
    Parameters:
    table_name (str): Name of the table to fetch data from
    output_file (str): Name of the CSV file to save data to
    """
    import io
    # Fetch data from the table
    response = supabase.table(table_name).select("*").execute()
    
    # Check if there's data
    if not response.data:
        print(f"No data found in table '{table_name}'")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(response.data)
    s = io.StringIO()
    
    # Save to CSV
    df.to_csv(s, index=False)
    return s.getvalue()