import logging
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from rich.logging import RichHandler
import supabase
from dotenv import load_dotenv
from rag import query_rag

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
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/query/{query}")
async def query_rag_endpoint(query: str) -> StreamingResponse:
    return StreamingResponse(query_rag(query), media_type="text/event-stream")

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )
        
        file_content = await file.read()
        
        filename = file.filename
        
        logger.info(f"Uploading file: {filename} to bucket: {bucket_name}")
        
        response = supabase_client.storage.from_(bucket_name).upload(
            path=filename,
            file=file_content,
            file_options={"content-type": "application/pdf"}
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": filename
            }
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