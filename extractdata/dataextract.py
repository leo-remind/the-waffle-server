from datetime import datetime
from pathlib import Path
import time
import pandas as pd
from dotenv import load_dotenv
import anthropic
import os
import base64
import psycopg2
from rich import print
import supabase
from pathvalidate import sanitize_filepath
from pinecone import Pinecone

from utils import (
    calculate_cost,
    convert_response_to_df,
    get_20_random_string,
    get_command_from,
    save_to_supabase_and_pinecone,
)

from prompts import EXTRACT_PROMPT

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


load_dotenv()


def claude_powered_extraction(
    image_data: bytes, client: anthropic.Anthropic
) -> dict[str, any]:
    """
    Returns a data frame given image data bytes. It will use an LLM to extract the data.
    """
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    media_type = (
        "image/png" if file_path.endswith("png") else "image/jpeg"
    )  # @HACK: rudimentary check for media type

    model_name = "claude-3-7-sonnet-20250219"

    st = time.time()
    message = client.messages.create(
        model=model_name,
        max_tokens=20000,
        temperature=1,
        system=EXTRACT_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_image,
                        },
                    }
                ],
            }
        ],
    )
    et = time.time()

    df = convert_response_to_df(message.content)

    token_counts = message.usage

    return {
        "data": df,
        "time_taken": et - st,
        "tokens_used": token_counts.input_tokens + token_counts.output_tokens,
        "approx_cost": calculate_cost(
            token_counts.input_tokens, token_counts.output_tokens, model_name=model_name
        ),
    }


def get_claude_powered_req(image: bytes, custom_id):
    """
    CLAUDE POWERED REQ
    """
    encoded_image = base64.b64encode(image).decode("utf-8")
    media_type = "image/png"  # @TODO: THIS ASSUMPTION IS FKED

    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model="claude-3-7-sonnet-20250219",
            max_tokens=20_000,
            system=EXTRACT_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded_image,
                            },
                        }
                    ],
                },
            ],
        ),
    )


def batched_system(
    image_datas: list[tuple[int, bytes]], pdf_name: str, client: anthropic.Anthropic
) -> any:
    """
    Dispatch the Claude-powered batched system for extracting data from images.

    Returns the message_batch object.
    """
    requests = []

    for ident, (page_no, image) in enumerate(image_datas):
        requests.append(
            get_claude_powered_req(image, custom_id=f"IMAGE_REQ_{ident}_pg{page_no}")
        )

    message_batch = client.messages.batches.create(
        requests=requests,
    )

    return message_batch


def synchronous_batched_system(
    image_datas: list[tuple[int, bytes]],
    pdf_name: str,
    pdf_supabase_url: str,
    client: anthropic.Anthropic,
    POLLING_RATE=12,
) -> any:
    """
    Dispatch several batch requests, and then wait until all the requests are complete.

    Returns the final results.
    """
    requests = []
    customId2Page = {}

    for ident, (page_no, image) in enumerate(image_datas):
        custom_id = f"IMAGE_REQ_{ident}_pg{page_no}"
        requests.append(get_claude_powered_req(image, custom_id=custom_id))
        customId2Page[custom_id] = page_no

    message_batch = client.messages.batches.create(
        requests=requests,
    )

    batch_id = message_batch.id

    print(f"Waiting for {batch_id} to complete...")

    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        if message_batch.processing_status == "ended":
            print(
                f"Time taken: {(message_batch.ended_at - message_batch.created_at).total_seconds()}s\n"
            )
            break
        print(f"[{datetime.now()}] Batch {batch_id} is still processing...")
        time.sleep(POLLING_RATE)

    final_results = []
    for result in client.messages.batches.results(
        batch_id,
    ):
        match result.result.type:
            case "succeeded":
                # print(f"Success! {result.custom_id}")
                final_results.append(result)

    converted_results = []
    if final_results:
        for result in final_results:
            try:
                df_results = convert_response_to_df(result.result.message.content)
            except:
                print("Failed to convert response to DF")
                continue

            statsmeta = {}
            statsmeta["input_tokens_used"] = result.result.message.usage.input_tokens
            statsmeta["output_tokens_used"] = result.result.message.usage.output_tokens
            statsmeta["approx_cost"] = calculate_cost(
                result.result.message.usage.input_tokens,
                result.result.message.usage.output_tokens,
                model_name="claude-3-7-sonnet-20250219-batched",
            )
            statsmeta["result_id"] = result.custom_id
            print(statsmeta)

            converted_results.append(
                {
                    "tables": df_results,
                    "stats": statsmeta,
                    "pdf_name": pdf_name,
                    "pdf_url": pdf_supabase_url,
                    "page_number": customId2Page[result.custom_id],
                }
            )

    else:
        raise ValueError("No results found")

    return converted_results


def load_image_data(file_path):
    with open(file_path, "rb") as f:
        image_data = f.read()
    return image_data


def wait_for_batched_to_complete(batch_id, pdf_name: str, client, POLLING_RATE=12):
    print(f"Waiting for {batch_id} to complete...")
    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        if message_batch.processing_status == "ended":
            print(
                f"Time taken: {(message_batch.ended_at - message_batch.created_at).total_seconds()}s\n"
            )
            break
        print(f"[{datetime.now()}] Batch {batch_id} is still processing...")
        time.sleep(POLLING_RATE)

    final_results = []
    for result in client.messages.batches.results(
        batch_id,
    ):
        match result.result.type:
            case "succeeded":
                # print(f"Success! {result.custom_id}")
                final_results.append(result)

    converted_results = []
    if final_results:
        for result in final_results:
            try:
                df_results = convert_response_to_df(result.result.message.content)
            except:
                print("Failed to convert response to DF")
                continue

            statsmeta = {}
            statsmeta["input_tokens_used"] = result.result.message.usage.input_tokens
            statsmeta["output_tokens_used"] = result.result.message.usage.output_tokens
            statsmeta["approx_cost"] = calculate_cost(
                result.result.message.usage.input_tokens,
                result.result.message.usage.output_tokens,
                model_name="claude-3-7-sonnet-20250219-batched",
            )
            statsmeta["result_id"] = result.custom_id
            print(statsmeta)

            converted_results.append(
                {
                    "tables": df_results,
                    "stats": statsmeta,
                    "pdf_title": pdf_name,
                    "page_number": result.result.message.content[0].page_number,
                }
            )
    else:
        raise ValueError("No results found")

    return converted_results


if __name__ == "__main__":


    supabase_client = psycopg2.connect(
        database=os.environ.get("PG_DBNAME"),
        user=os.environ.get("PG_USER"),
        password=os.environ.get("PG_PASSWORD"),
        host=os.environ.get("PG_HOST"),
        port=os.environ.get("PG_PORT"),
        sslmode="require",
    )
    print("Connected to Supabase")
    anthropic_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    file_paths = list(Path("pngs").glob("*.png"))
    # file_paths = ["pngs/notitle.png"]

    print(f"Processing {len(file_paths)} images...")

    image_datas = map(load_image_data, file_paths)
    image_datas = list(enumerate(image_datas))  # sample page numbers

    table_responses = synchronous_batched_system(
        image_datas=image_datas,
        pdf_name="test.pdf",
        pdf_supabase_url="https://thinklude.ai",
        client=anthropic_client,
    )
    save_to_supabase_and_pinecone(table_responses, supabase_client, pinecone_client)
