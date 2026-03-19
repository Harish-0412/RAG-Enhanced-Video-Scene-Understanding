from __future__ import annotations
import os
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Import existing pipeline phases
from src.phase1_sampling import process_video as phase1_process
from src.phase2_visual import run_phase2_visual as phase2_enrich
from src.phase3_indexing import create_index as phase3_index
from src.phase4_rag import VideoRAG

app = FastAPI(title="VideoSceneRAG API")

# Enable CORS for all origins with credentials support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage configuration
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount data directory for serving processed frames/metadata if needed
app.mount("/data", StaticFiles(directory="data"), name="data")

# Global state to track processing status
processing_status: Dict[str, Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    video_id: str
    query: str

class QueryResponse(BaseModel):
    answer: str
    timestamp: float
    confidence: float
    citations: List[Dict[str, Any]]

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    # Keep the original extension
    orig_ext = Path(file.filename).suffix
    video_filename = f"{video_id}{orig_ext}"
    file_path = UPLOAD_DIR / video_filename
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processing_status[video_id] = {
        "status": "uploaded", 
        "progress": 0, 
        "filename": file.filename,
        "extension": orig_ext.replace(".", ""),
        "path": str(file_path)
    }
    
    # Start background processing
    background_tasks.add_task(process_pipeline, video_id, str(file_path))
    
    return {"video_id": video_id, "filename": file.filename, "extension": orig_ext.replace(".", "")}

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")
    return processing_status[video_id]

async def process_pipeline(video_id: str, video_path: str):
    try:
        # Phase 1: Sampling & Scene Detection
        processing_status[video_id]["status"] = "Phase 1: Scene Detection"
        processing_status[video_id]["progress"] = 10
        # We need to run this in a thread pool since it's CPU intensive
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, phase1_process, video_path)
        
        # Phase 2: Visual Enrichment
        processing_status[video_id]["status"] = "Phase 2: Visual Analysis (Gemini)"
        processing_status[video_id]["progress"] = 40
        await loop.run_in_executor(None, phase2_enrich)
        
        # Phase 3: Indexing
        processing_status[video_id]["status"] = "Phase 3: Building Vector Index"
        processing_status[video_id]["progress"] = 80
        await loop.run_in_executor(None, phase3_index)
        
        processing_status[video_id]["status"] = "completed"
        processing_status[video_id]["progress"] = 100
        logger.info(f"Successfully processed video {video_id}")
        
    except Exception as e:
        processing_status[video_id]["status"] = "failed"
        processing_status[video_id]["error"] = str(e)
        logger.error(f"Error processing {video_id}: {e}", exc_info=True)

# Global RAG instance
rag_engine: Optional[VideoRAG] = None

def get_rag():
    global rag_engine
    if rag_engine is None:
        try:
            rag_engine = VideoRAG()
        except Exception as e:
            logger.error(f"Failed to initialize VideoRAG: {e}")
            raise e
    return rag_engine

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # Use the singleton RAG instance
        rag = get_rag()
        result = rag.ask(request.query)
        
        # Extract all timestamps from citations for comprehensive references
        timestamps = []
        if result.get("citations"):
            for citation in result["citations"]:
                if "start_seconds" in citation and citation["start_seconds"] is not None:
                    timestamp = float(citation["start_seconds"])
                    ts_str = citation.get("timestamp", str(timestamp))
                    timestamps.append({
                        "timestamp": timestamp,
                        "formatted": ts_str,
                        "context": citation.get("visual_summary", "")[:100] + "..." if citation.get("visual_summary") else ""
                    })
                elif "timestamp" in citation:
                    # Parse timestamp like "00:01:23" to seconds
                    ts_str = citation["timestamp"]
                    try:
                        parts = ts_str.split(':')
                        if len(parts) == 3:
                            timestamp = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                        elif len(parts) == 2:
                            timestamp = int(parts[0]) * 60 + int(parts[1])
                        else:
                            timestamp = float(ts_str)
                        timestamps.append({
                            "timestamp": timestamp,
                            "formatted": ts_str,
                            "context": citation.get("visual_summary", "")[:100] + "..." if citation.get("visual_summary") else ""
                        })
                    except:
                        continue
        
        # Use the first timestamp for video jumping, but return all for frontend
        primary_timestamp = timestamps[0]["timestamp"] if timestamps else 0.0
        
        return QueryResponse(
            answer=result.get("answer", "❌  Retriever unavailable or processing failed."),
            timestamp=primary_timestamp,
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", [])
        )
    except Exception as e:
        logger.error(f"RAG error in /ask: {e}", exc_info=True)
        # Return a valid QueryResponse even on internal error to avoid 500
        return QueryResponse(
            answer=f"❌  Error: {str(e)}",
            timestamp=0.0,
            confidence=0.0,
            citations=[]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
