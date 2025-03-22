import logging
import os
from asyncio import sleep
from typing import List

from fastapi import WebSocket
import psycopg2
from dotenv import load_dotenv
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
log = logging.getLogger("RAG")

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

stream_fmt = lambda a: f"data: {a}\n\n"


async def query_rag(websocket: WebSocket, og_query: str, verbose: bool = False, graph: bool = False):
    await websocket.send_text("stream: Finding most relevant tables\n\n")
    query = query_augmentation(og_query)
    tables = find_k_relevant_tables(query)
    await websocket.send_text(f"stream: Generating response from {len(tables)} relevant table/s\n\n")
    await sleep(0.1)
    schemas = get_table_schemas(tables)
    schema_str = "\n\n".join([f"-- table topic: {table['text']}\n{schema}" for table, schema in zip(tables, schemas)])
    sql_query = generate_sql_query(query, schema_str)
    await websocket.send_text(f"stream: Querying SQL Database with query\n\n")
    await sleep(0.1)
    results = execute_sql_query(sql_query)
    await websocket.send_text("stream: Values collated, generating response ...")
    await sleep(0.1)
    response = generate_response(
        og_query, results, sql_query, schema_str, verbose=verbose, graph=graph
    )
    await websocket.send_text(f"kill: {response}")
    await sleep(0.1)
    log.info(response)


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
    log.info(pretty_repr(top_k_tables))

    return top_k_tables


def query_augmentation(query: str, llm: ChatOpenAI = chatgpt_4o) -> str:
    chain = QUERY_AUGMENTATION_PROMPT | llm

    response = chain.invoke(query)
    log.info(f"augmented query: {response.content}")
    return response.content


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
    log.info(schemas)
    return schemas


def generate_sql_query(query: str, schema_str: str, llm: ChatOpenAI = chatgpt_o3_mini):
    chain = SQL_GENERATION_PROMPT | llm
    log.info(f"query: {query}")

    response = chain.invoke({"query": query, "schema": schema_str})
    log.info(f"generated_response: {response.content}")

    return response.content


def execute_sql_query(sql_query: str):
    global pg_conn

    sql_query = sql_query.strip("`").removeprefix("sql")
    cursor = pg_conn.cursor()
    cursor.execute(sql_query)

    results = cursor.fetchall()

    cursor.close()
    log.info(f"Executed query, results: {pretty_repr(results)}")

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
