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

from constants import RETRIEVAL_THRESHOLD
from prompts import (
    NORMAL_RESPONSE_PROMPT,
    QUERY_AUGMENTATION_PROMPT,
    SQL_GENERATION_PROMPT,
)

load_dotenv()
logger = logging.getLogger(__file__)

MODEL_NAME_RE = re.compile(r"SELECT.*?FROM\s*(.{20}).*", re.IGNORECASE)

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


def stream_fmt(a):
    return f"data: {a}\n\n"


# stream_fmt = lambda a: f"data: {a}\n\n"


async def query_rag(
    websocket: WebSocket, og_query: str, verbose: bool = False, graph: bool = False
):
    await websocket.send_text(
        json.dumps({"isStreaming": True, "message": "Finding most relevant tables\n\n"})
    )
    query = query_augmentation(og_query)
    tables = find_k_relevant_tables(query)
    await websocket.send_text(
        json.dumps(
            {
                "isStreaming": True,
                "message": f"Generating response from {len(tables)} relevant table/s\n\n",
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
    sql_query = generate_sql_query(query, schema_str)
    await websocket.send_text(
        json.dumps({"isStreaming": True, "message": "Querying SQL Database with query"})
    )
    await sleep(0.1)
    results = execute_sql_query(sql_query)

    await websocket.send_text(
        json.dumps(
            {"isStreaming": True, "message": "Values collated, generating response..."}
        )
    )
    await sleep(0.1)
    response = generate_response(
        og_query, results, sql_query, schema_str, verbose=verbose, graph=graph
    )
    model_name_match = MODEL_NAME_RE.match(sql_query)
    if model_name_match:
        primary_table = model_name_match.groups()[0]
    else:
        primary_table = ""

    await websocket.send_text(
        json.dumps(
            {
                "isStreaming": False,
                "message": response,
                "primaryTable": primary_table,
                "tables": [
                    get_table_as_json(table["supabase_table_name"]) for table in tables
                ],
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


def find_k_relevant_tables(query: str, top_k: int = 3) -> List[str]:
    index = pc.Index("the-waffle")

    results = index.search_records(
        namespace="",
        query={"inputs": {"text": query}, "top_k": top_k * 4},
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
    log.info(f"generated_response: {response.content}")
    sql_query = extract_first_code_block(response.content)
    log.info(f"generated sql query:\n {sql_query}")

    return sql_query


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
    sql_query: str,
    schema_str: str,
    llm: ChatOpenAI = chatgpt_o3_mini,
    verbose: bool = False,
    graph: bool = False,
):
    prompt = NORMAL_RESPONSE_PROMPT  # VERBOSE_RESPONSE_PROMPT if verbose else NORMAL_RESPONSE_PROMPT
    chain = prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "result": str(results),
            "sql_query": sql_query,
            "schema": schema_str,
        }
    )

    return response.content


if __name__ == "__main__":
    execute_sql_query('SELECT * FROM "METADATA"')
