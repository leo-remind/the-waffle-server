import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger

import numpy as np
import pandas as pd
import psycopg2
from openai import OpenAI

from .prompts import QUICK_FIX_PROMPT

logger = getLogger(__file__)

N_THREADS = 10


def validate_json(json_data):
    try:
        json.loads(json_data)
    except ValueError as _:
        return False
    return True


def fix_len_using_chatgpt(inp: dict, actual_len, broken_len) -> pd.DataFrame:
    """
    Fix the length of the dataframe using ChatGPT.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    txt_dmp = json.dumps(inp)
    st = time.time()
    myprompt = (
        QUICK_FIX_PROMPT.replace("{{CORRECT_LENGTH}}", str(actual_len))
        .replace("{{CURRENT_LENGTH}}", str(broken_len))
        .replace("{{DATA}}", txt_dmp)
    )
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": myprompt}],
            },
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )
    et = time.time()
    # logger.info(response.usage)
    resp = response.output[0].content[0].text
    # logger.info(resp)
    if "```" in resp:
        resp = resp.strip("```")
        resp = resp.strip("json")

    # logger.info(resp)

    if not validate_json(resp):
        raise ValueError("Invalid JSON yet again")

    logger.info(
        f"FL_QUICK took {et - st}s and {response.usage.total_tokens}tokens, costing ${calculate_cost(response.usage.input_tokens, response.usage.output_tokens, 'gpt-4o')}"
    )
    return json.loads(resp)


def convert_response_to_df(message_content: list) -> list[pd.DataFrame]:
    """
    Converts claude response to a pandas dataframe.
    """
    try:
        text_block = message_content[0]
        try:
            data = json.loads(text_block.text)
        except TypeError:
            data = json.loads(text_block["text"])

        ret_tups = []
        for conv_data in data:
            if not isinstance(conv_data, dict):
                raise ValueError("Invalid response")
            # logger.info(conv_data)
            title = conv_data.get("title")
            min_year = conv_data.get("min_year")
            max_year = conv_data.get("max_year")
            data = conv_data.get("data")

            all_lens = [len(v) for v in data.values()]
            if len(set(all_lens)) != 1:
                # most common length
                base_len = max(set(all_lens), key=all_lens.count)
                logger.info(
                    f"[CONVERTOR] Mismatched lengths! Most common len: {base_len}"
                )
                for k, v in data.items():
                    if len(v) != base_len:
                        logger.info(f"[red]Warning:[/red] {k} has length {len(v)}")
                        # logger.info(f"ChatGPT Fast Query: {k}:{v}")
                        inp = {k: v}
                        fixed_col = fix_len_using_chatgpt(inp, base_len, len(v))
                        logger.info(f"Fixed column: {fixed_col}\n")
                        data[k] = fixed_col.get(k)

            # if still not equal, then we have a problem
            all_lens = [len(v) for v in data.values()]
            if len(set(all_lens)) != 1:
                # manually fix by adding "NONE" values to the end of whicever columns are small until we match the longest column
                base_len = max([len(v) for v in data.values()])
                for k, v in data.items():
                    if len(v) != base_len:
                        logger.info(f"[red]Warning:[/red] {k} has length {len(v)}")
                        data[k] = data[k] + ["NONE"] * (base_len - len(v))

            df = pd.DataFrame(conv_data.get("data"))

            ret_tups.append(
                {"title": title, "min_year": min_year, "max_year": max_year, "df": df}
            )
    except Exception as e:
        logger.info("Invalid response: {}".format(e))
        raise ValueError("Invalid response")
    return ret_tups


def calculate_cost(
    input_tokens, output_tokens, model_name="claude-3-7-sonnet-20250219"
):
    """
    Calculates the cost of the API call.
    """

    match model_name:
        case "claude-3-7-sonnet-20250219":
            input_cost_per1m = 3
            output_cost_per1m = 15
        case "claude-3-7-sonnet-20250219-batched":
            input_cost_per1m = 1.5
            output_cost_per1m = 7.5
        case "gpt-4o-mini":
            input_cost_per1m = 0.15
            output_cost_per1m = 0.60
        case "gpt-4o":
            input_cost_per1m = 2.50
            output_cost_per1m = 10.00
        case _:
            raise ValueError("Model not supported")

    return (
        input_tokens * input_cost_per1m + output_tokens * output_cost_per1m
    ) / 1_000_000


postgressql_type_map = {
    np.dtype("int64"): "INTEGER",
    np.dtype("float64"): "REAL",
    np.dtype("object"): "TEXT",
    np.dtype("datetime64"): "TIMESTAMP",
    np.dtype("bool"): "BOOLEAN",
}


def get_command_from(df: pd.DataFrame, title) -> dict:
    """
    Get the schema from the dataframe.
    """

    # replace "NONE" with None values in the dataframe
    df = df.replace("NONE", None)

    # now infer
    df = df.infer_objects()
    # logger.info(df.dtypes)
    # convert to sql_schema in the form col_name TYPE, col_name TYPE, ...

    # sql_schema = ", ".join(
    #     [f"{col} {postgressql_type_map.get(df[col].dtype, "TEXT")}" for col in df.columns]
    # )

    sql_command = pd.io.sql.get_schema(df.reset_index(), title)

    insert_command = f"""INSERT INTO "{title}" ({", ".join(['"' + col + '"' for col in df.columns])}) VALUES 
    ({", ".join(["%s" for _ in df.columns])});"""
    sql_command = sql_command.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
    sql_command = sql_command.replace('"index" INTEGER', '"index" SERIAL PRIMARY KEY')

    return sql_command, insert_command, df


def get_20_random_string():
    """
    Get a random string of length 20.
    """
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=20))

def comment_the_schema(schema, df):
    """
    Comment the schema with the column names.
    """
    x = schema.split("\n")[1:]

    for i, col in enumerate(df.columns):
        x[i] = f"{x[i]} -- {', '.join(random.choices(list(set(col.values)), k=3))}"
    schema = "\n".join(x)

    return schema

def save_single_to_supabase_and_pinecone(response, supabase_client, pinecone_client):
    table_data = response["tables"]
    pc_upsert = []

    for table in table_data:
        cur = supabase_client.cursor()

        logger.info(("=" * 80) + "\n\n")
        logger.info(
            f"Table: '{table['title']}', Year Range: ({table['min_year']}-{table['max_year']})"
        )

        try:
            random_string = get_20_random_string()

            schema, insert_command, typed_df = get_command_from(
                df=table["df"], title=random_string
            )
            commented_schema = comment_the_schema(schema, table["df"])
            logger.info(f"Commented schema: {commented_schema}")
            typed_df = typed_df.replace({np.nan: None, "NONE": None})
            typed_df = typed_df.replace({None: "NULL"})
            command = schema
            logger.info(f"\n[bold]{command}[/bold]\n")
            cur.execute(command)
            logger.info(f"\n[bold]{insert_command}[/bold]\n")
            c = 1
            for row in typed_df.itertuples(index=False, name=None):
                row_f = []
                for r in row:
                    if r == "NULL":
                        row_f.append(None)
                    else:
                        row_f.append(str(r))

                # logger.info(f"{c}: {row_f} with {len(row_f)}")

                # logger.info(insert_command % tuple(row_f))

                cur.execute(insert_command, tuple(row_f))
                c += 1

            # add to metadata
            execu = (
                'INSERT INTO "METADATA"'
                + " (schema, table_heading, supabase_table_name, min_year, max_year, pdf_title, pdf_url, page_number)"
                + " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            )

            cur.execute(
                execu,
                (
                    commented_schema,
                    table["title"],
                    random_string,
                    int(table["min_year"]) if table["min_year"] != "" else 0,
                    int(table["max_year"]) if table["max_year"] != "" else 3000,
                    response["pdf_name"],
                    response["pdf_url"],
                    int(response["page_number"]),
                ),
            )

            pc_upsert.append(
                {
                    "id": random_string,
                    "text": table["title"],
                    "table_heading": table["title"],
                    "min_year": table["min_year"],
                    "max_year": table["max_year"],
                    "supabase_table_name": random_string,
                    "pdf_name": response["pdf_name"],
                }
            )

        except (Exception, psycopg2.DatabaseError) as error:
            logger.info(f"ERROR: {error}")
            cur.execute("ROLLBACK")
            raise error
        
        finally:
            cur.close()
            supabase_client.commit()
    try:
        index = pinecone_client.Index("the-waffle")

        index.upsert_records("", pc_upsert)
    except Exception as e:
        logger.info(f"Failed to upsert to Pinecone: {e}")
        raise e


def save_to_supabase_and_pinecone(table_responses, supabase_client, pinecone_client):
    logger.info(f"started thread pool with {N_THREADS}")
    with ThreadPoolExecutor(N_THREADS) as exe:

        futures = {exe.submit(save_single_to_supabase_and_pinecone, x, supabase_client, pinecone_client): x for x in table_responses}

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Task {futures[future]} generated an exception: {e}")
