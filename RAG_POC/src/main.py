import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Import the necessary classes from the main program
from src.RAG import HuggingFaceRAGPipeline, process_files

# Create FastAPI app
app = FastAPI(title="RAG Document Query API")

# Mount static files
app.mount("/static",StaticFiles(directory="templates"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("templates/index.html")


# Configuration model for RAG pipeline
class RAGConfig(BaseModel):
    hf_token: Optional[str] = None
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    qa_model: str = 'deepset/roberta-base-squad2'

# Query model
class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    max_answer_length: int = 50

# Result model
class QueryResponse(BaseModel):
    answer: str
    context: str
    contexts: List[dict]

# Global RAG pipeline 
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    # Get Hugging Face token from environment variable
    HF_TOKEN = os.getenv('HF_TOKEN')
    
    if not HF_TOKEN:
        raise ValueError("Hugging Face API token is required. Set HF_TOKEN environment variable.")
    
    rag_pipeline = HuggingFaceRAGPipeline(
        hf_token=HF_TOKEN
    )

@app.post("/upload-documents/", response_model=str)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents and preprocess them
    
    Args:
        files: List of uploaded files
    
    Returns:
        Confirmation message
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files temporarily
    file_paths = []
    try:
        for file in files:
            # Generate a unique temporary file path using tempfile
            temp_path = os.path.join(temp_dir, file.filename)
            
            # Save the uploaded file
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            
            file_paths.append(temp_path)
        
        # Process files and preprocess documents
        documents = process_files(file_paths)
        rag_pipeline.preprocess_documents(documents)
        return "Documents uploaded and preprocessed successfully"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Optional: Clean up temporary files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the preprocessed documents
    
    Args:
        request: Query request with optional context
    
    Returns:
        Query response with answer and context
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=400, detail="No documents preprocessed. Upload documents first.")
    
    try:
        result = rag_pipeline.answer_question(
            query=request.query, 
            context=request.context, 
            max_answer_length=request.max_answer_length
        )
        
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example of how to run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)