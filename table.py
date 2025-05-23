import io
import os
from logging import getLogger

import fitz
import numpy as np
import psycopg2
import torch
from anthropic import Anthropic
from PIL import Image
from pinecone import Pinecone
from transformers import DetrFeatureExtractor, TableTransformerForObjectDetection

from constants import LOAD_MODEL
from extractdata.dataextract import asynchronous_batched_system
from extractdata.utils import save_to_supabase_and_pinecone

logger = getLogger(__file__)

if LOAD_MODEL:
    feature_extractor = DetrFeatureExtractor()
    detection_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection"
    )

supabase_client = psycopg2.connect(
    database=os.environ.get("PG_DBNAME"),
    user=os.environ.get("PG_USER"),
    password=os.environ.get("PG_PASSWORD"),
    host=os.environ.get("PG_HOST"),
    port=os.environ.get("PG_PORT"),
    sslmode="require",
)

anthropic_client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

logger.debug("initialized feature extractor, detection model and anthropic client")

WHITE_PCT_THRESHOLD = 96.0


def pdf_to_images(pdf: bytes, dpi: int = 300) -> list[Image]:
    """
    Convert PDF bytes to a list of PIL Image objects

    ### Params
        pdf_bytes (bytes): The PDF file as bytes
        dpi (int): dpi

    ### Returns
        list: List of PIL Image objects, one for each page
    """
    pdf_stream = io.BytesIO(pdf)
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        zoom = dpi / 72  # 72 is the base DPI
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img_bytes = pix.tobytes("png")
        img_stream = io.BytesIO(img_bytes)
        img = Image.open(img_stream)

        images.append(img)

    pdf_document.close()
    return images


async def process_pdf(pdf: bytes, filename: str, url: str):
    """
    Process a pdf

    ### Params
        pdf (bytes): the bytes of a pdf file
        filename (str): name of the pdf
        url (str): url of the pdf
    """
    global anthropic_client, detection_model, feature_extractor

    with torch.no_grad():
        logger.debug("started processing pdf")
        images_with_tables = filter_pages_with_tables(
            detection_model, feature_extractor, pdf_to_images(pdf)
        )

    if not images_with_tables:
        logger.debug("pdf has no pages with tables")

    logger.debug("sent images with tables to pjr")

    table_responses = await asynchronous_batched_system(
        image_datas=images_with_tables,
        pdf_name=filename,
        pdf_supabase_url=url,
        client=anthropic_client,
    )
    logger.info("received table responses from sync batched system")
    save_to_supabase_and_pinecone(table_responses, supabase_client, pinecone_client)


def calculate_white_percentage(image: Image, threshold: int = 245) -> float:
    """
    Calculate the percentage of white pixels in an PIL Image.

    ### Params
        image (PIL.Image): The input image
        threshold (int, optional): Pixel value threshold to consider as white (0-255).

    ### Returns
        float: Percentage of white pixels (0.0 to 100.0)
    """
    if image.mode != "L":
        image = image.convert("L")

    img_array = np.array(image)
    white_pixels = np.sum(img_array >= threshold)
    total_pixels = img_array.size

    return (white_pixels / total_pixels) * 100.0


def _is_page_valid(
    score: float,
    white_pct: float,
    threshold_if_white: float = 0.9,
    threshold_if_not_white: float = 0.8,
) -> bool:
    return (
        score > threshold_if_white
        if white_pct > WHITE_PCT_THRESHOLD
        else score > threshold_if_not_white
    )


def filter_pages_with_tables(
    model: TableTransformerForObjectDetection,
    feature_extractor: DetrFeatureExtractor,
    images: list[Image],
    threshold: float = 0.7,
) -> list[tuple[int, Image]]:
    """
    returns a list of images that have tables in them

    ### Params
        model (TableTransformerForObjectDetection): table transformer model
        feature_extractor (DetrFeatureExtractor): feature extractor
        images (list[Image]): the list of images to process
        threshold (float, default = 0.7): threshold passed to `feature_extractor.post_process_object_detection`

    ### Returns
        list[tuple[int, Image]]: a list of `page_no`, `PIL.Image` that have tables in them
    """
    with_tables = []
    for no, image in enumerate(images, 1):
        # convert to grayscale wihout losing the dimensions
        image_original = image.copy()
        image = image.convert("L").convert("RGB")
        white_pct = calculate_white_percentage(image)

        # dilation
        # if white_pct > 50:
        #     page = page.filter(ImageFilter.MinFilter(4))
        encoding = feature_extractor(image, return_tensors="pt")

        output = model(**encoding)
        width, height = image.size
        result = feature_extractor.post_process_object_detection(
            output, threshold=threshold, target_sizes=[(height, width)]
        )[0]

        if (
            len(
                list(
                    filter(
                        lambda score: _is_page_valid(score, white_pct), result["scores"]
                    )
                )
            )
            == 0
        ):
            logger.debug(
                f"\t{no}: no tables, scores = {', '.join('%.2f' % i for i in result['scores'])}, white pct = {white_pct}"
            )
            continue

        logger.debug(
            f"\t{no}: has {len(result['scores'])} tables, scores = {', '.join('%.2f' % i for i in result['scores'])}, white pct = {white_pct}"
        )
        with_tables.append((no, image_original))

    return with_tables


if __name__ == "__main__":
    from sys import argv

    with open(argv[1], "rb") as f:
        pdf = f.read()
    process_pdf(pdf)
