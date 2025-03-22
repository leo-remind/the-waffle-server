import asyncio
import base64
import io
import os
import time
from datetime import datetime
from logging import getLogger
from pathlib import Path

import anthropic
import numpy as np
import psycopg2
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv
from PIL import Image
from pinecone import Pinecone

from .prompts import EXTRACT_PROMPT
from .utils import (
    calculate_cost,
    convert_response_to_df,
    save_to_supabase_and_pinecone,
)

logger = getLogger(__file__)
load_dotenv()


def otsu_threshold(img: Image) -> Image:
    img_array = np.array(img.convert("L"))
    histogram, bin_edges = np.histogram(img_array, bins=256, range=(0, 256))
    total_pixels = img_array.size
    cumsum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * np.arange(256)) / (cumsum + 1e-10)
    global_mean = np.sum(histogram * np.arange(256)) / total_pixels
    between_class_variances = np.zeros(256)

    for t in range(256):
        w_bg = cumsum[t] / total_pixels
        w_fg = 1 - w_bg

        if w_bg > 0 and w_fg > 0:
            mean_bg = cumulative_mean[t]
            mean_fg = (global_mean * total_pixels - mean_bg * cumsum[t]) / (
                total_pixels - cumsum[t]
            )

            between_class_variances[t] = w_bg * w_fg * (mean_bg - mean_fg) ** 2

    threshold = np.argmax(between_class_variances)

    return Image.fromarray(np.where(img_array > threshold, 255, 0).astype(np.uint8))


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


async def asynchronous_batched_system(
    image_datas: list[tuple[int, Image]],
    pdf_name: str,
    pdf_supabase_url: str,
    client: anthropic.Anthropic,
    POLLING_RATE=10,
) -> any:
    """
    Dispatch several batch requests, and then wait until all the requests are complete.

    Returns the final results.
    """
    requests = []
    customId2Page = {}

    logger.info(f"image_datas has {len(image_datas)} images")
    for ident, (page_no, image) in enumerate(image_datas):
        bytes_stream = io.BytesIO()
        custom_id = f"IMAGE_REQ_{ident}_pg{page_no}"
        # image = otsu_threshold(image)
        image.save(bytes_stream, format="png")
        # save image to disk
        # image.save(f"otsu_{custom_id}.png", format="png")

        bytes_stream.seek(0)
        requests.append(
            get_claude_powered_req(bytes_stream.read(), custom_id=custom_id)
        )
        customId2Page[custom_id] = page_no
        bytes_stream.close()

    message_batch = client.messages.batches.create(
        requests=requests,
    )
    batch_id = message_batch.id
    logger.info(f"Waiting for {batch_id} to complete...")

    while True:
        message_batch = client.messages.batches.retrieve(batch_id)
        if message_batch.processing_status == "ended":
            logger.info(
                f"Time taken: {(message_batch.ended_at - message_batch.created_at).total_seconds()}s\n"
            )
            break
        logger.info(f"[{datetime.now()}] Batch {batch_id} is still processing...")
        await asyncio.sleep(POLLING_RATE)

    final_results = []
    for result in client.messages.batches.results(
        batch_id,
    ):
        match result.result.type:
            case "succeeded":
                logger.info(f"Success! {result.custom_id}")
                final_results.append(result)
            case default:
                logger.info(f"Not Success: {result.custom_id} {default}")

    converted_results = []
    if final_results:
        ec = 0
        for result in final_results:
            try:
                df_results = convert_response_to_df(result.result.message.content)
                if df_results is None:
                    continue # skip this result
            except Exception as e:
                logger.info(f"Failed to convert response to DF: {e}")
                ec += 1
                raise e

            statsmeta = {}
            statsmeta["input_tokens_used"] = result.result.message.usage.input_tokens
            statsmeta["output_tokens_used"] = result.result.message.usage.output_tokens
            statsmeta["approx_cost"] = calculate_cost(
                result.result.message.usage.input_tokens,
                result.result.message.usage.output_tokens,
                model_name="claude-3-7-sonnet-20250219-batched",
            )
            statsmeta["result_id"] = result.custom_id
            logger.info(f"statsmeta: {statsmeta}")

            converted_results.append(
                {
                    "tables": df_results,
                    "stats": statsmeta,
                    "pdf_name": pdf_name,
                    "pdf_url": pdf_supabase_url,
                    "page_number": customId2Page[result.custom_id],
                }
            )

        if ec == len(final_results):
            raise ValueError("All results failed to convert")
    else:
        raise ValueError("No results found")

    logger.info("finished batch")
    return converted_results


def load_image_data(file_path):
    with open(file_path, "rb") as f:
        image_data = f.read()
    return image_data


if __name__ == "__main__":
    supabase_client = psycopg2.connect(
        database=os.environ.get("PG_DBNAME"),
        user=os.environ.get("PG_USER"),
        password=os.environ.get("PG_PASSWORD"),
        host=os.environ.get("PG_HOST"),
        port=os.environ.get("PG_PORT"),
        sslmode="require",
    )
    logger.info("Connected to Supabase")
    anthropic_client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    file_paths = list(Path("pngs").glob("*.png"))
    # file_paths = ["pngs/notitle.png"]

    logger.info(f"Processing {len(file_paths)} images...")

    image_datas = map(load_image_data, file_paths)
    image_datas = list(enumerate(image_datas))  # sample page numbers

    table_responses = synchronous_batched_system(
        image_datas=image_datas,
        pdf_name="test.pdf",
        pdf_supabase_url="https://thinklude.ai",
        client=anthropic_client,
    )
    save_to_supabase_and_pinecone(table_responses, supabase_client, pinecone_client)
