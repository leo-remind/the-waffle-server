import json
import logging
import os

import supabase
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from rich.logging import RichHandler

from rag import query_rag
from table import process_pdf

load_dotenv()

FORMAT = "%(message)s"
logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__file__)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_PVT_KEY")
bucket_name = os.getenv("SUPABASE_BUCKET_NAME")

# if bucket_name is None:
#     raise ValueError("SUPABASE_BUCKET_NAME environment variable is required")

if not supabase_url or not supabase_key:
    logger.error("Supabase credentials not found in environment variables")
    raise ValueError("Supabase credentials not configured")

supabase_client = supabase.create_client(supabase_url, supabase_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/query/ws")
async def query_rag_endpoint(websocket: WebSocket) -> StreamingResponse:
    await websocket.accept()
    query_data = json.loads(await websocket.receive_text())
    await query_rag(
        websocket,
        query_data["query"],
        verbose=query_data["verbose"],
        graph=query_data["graph"],
    )


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_content = await file.read()

        filename = file.filename

        logger.info(f"Uploading file: {filename} to bucket: {bucket_name}")

        response = supabase_client.storage.from_(bucket_name).upload(
            path=filename,
            file=file_content,
            file_options={"content-type": "application/pdf"},
        )

        url = supabase_client.storage.from_(bucket_name).get_public_url(response.path)

        try:
            process_pdf(file_content, filename, url)
        except Exception as e:
            return JSONResponse(
                status_code=500, content={"error": f"error in processing pdf {e}"}
            )

        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded successfully", "filename": filename},
        )

    except supabase.StorageException as e:
        logger.error(f"Supabase storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
