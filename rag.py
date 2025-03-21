import logging
import os
from typing import List

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


async def query_rag(og_query: str, verbose: bool = False, graph: bool = False):
    yield stream_fmt("Finding most relevant tables")
    query = query_augmentation(og_query)
    tables = find_k_relevant_tables(query)
    yield stream_fmt(f"Generating response from {len(tables)} relevant table/s")
    yield stream_fmt(str(tables))
    schemas = get_table_schemas(tables)
    sql_query = generate_sql_query(query, schemas)
    yield stream_fmt(f"Querying SQL Database with query")
    yield stream_fmt(str(sql_query))
    results = execute_sql_query(sql_query)
    yield stream_fmt("Values collated, generating response ...")
    response = generate_response(
        og_query, results, sql_query, schemas, verbose=verbose, graph=graph
    )
    yield stream_fmt(response)
    log.info(response)


def find_k_relevant_tables(query: str, top_k: int = 2) -> List[str]:
    index = pc.Index("the-waffle")

    results = index.search_records(
        namespace="",
        query={"inputs": {"text": query}, "top_k": top_k * 3},
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
    return schemas


def generate_sql_query(query: str, schemas: List, llm: ChatOpenAI = chatgpt_4o):
    chain = SQL_GENERATION_PROMPT | llm
    log.info(f"query: {query}")

    response = chain.invoke({"query": query, "schema": "\n\n".join(schemas)})
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
    schemas: List[str],
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
            "sql_query": sql_query,
            "schema": "\n\n".join(schemas),
        }
    )

    return response.content


if __name__ == "__main__":
    execute_sql_query('SELECT * FROM "METADATA"')
