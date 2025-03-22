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
from table import pdf_to_images, process_pdf

load_dotenv()

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("pdf_upload")

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
        pdf_name=query_data["pdf_name"],
    )


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_content = await file.read()
        filename = file.filename
        
        # Check if file with the same name already exists in the bucket
        try:
            # List files in the bucket
            file_list = supabase_client.storage.from_(bucket_name).list()
            
            # Check if the filename already exists
            file_exists = any(item["name"] == filename for item in file_list)
            
            if file_exists:
                logger.info(f"File {filename} already exists in bucket")
                # Get the public URL for the existing file
                url = supabase_client.storage.from_(bucket_name).get_public_url(filename)
                
                # Return a successful response since we have the file
                return JSONResponse(
                    status_code=200,
                    content={
                        "message": f"File '{filename}' already exists and is ready to use",
                        "filename": filename,
                        "url": url,
                        "exists": True
                    }
                )
                
        except Exception as e:
            logger.error(f"Error checking if file exists: {str(e)}")
            # Continue with upload if we couldn't check (fail open)
        
        # If we got here, the file doesn't exist or we couldn't check
        logger.info(f"Uploading new file: {filename} to bucket: {bucket_name}")

        response = supabase_client.storage.from_(bucket_name).upload(
            path=filename,
            file=file_content,
            file_options={"content-type": "application/pdf"},
        )

        url = supabase_client.storage.from_(bucket_name).get_public_url(response.path)
        
        # TODO: HAHHAHAHA TIME TO PROCESS THE PDF BUT I DONT WANT TO CANCER MY CLAUDE
        try:
            process_pdf(file_content, filename, url)
            logger.info(f"File {filename} processed successfully")
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return JSONResponse(
                status_code=500, content={"error": f"Error processing PDF: {e}"}
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": f"File '{filename}' uploaded and processed successfully", 
                "filename": filename,
                "url": url,
                "exists": False
            }
        )

    except supabase.StorageException as e:
        logger.error(f"Supabase storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/available-pdfs")
async def available_pdfs():
    try:
        response = supabase_client.storage.from_(bucket_name).list("")

        return JSONResponse(
            status_code=200,
            content={"files": [x for x in response if not x["name"].startswith(".")]},
        )

    except supabase.StorageException as e:
        logger.error(f"Supabase storage error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting available pdfs: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get available pdfs: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
