from fastapi import FastAPI, BackgroundTasks
from dotenv import load_dotenv
from app.models import VideoRequest
from app.processor import process_video_and_upsert

# Load environment variables from .env file on startup
load_dotenv()

app = FastAPI()

@app.post("/process-video")
async def create_processing_job(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Accepts a video URL and queues it for background processing.
    This ensures the HTTP request returns quickly without timing out.
    """
    background_tasks.add_task(process_video_and_upsert, str(request.video_url), request.video_id)
    return {"message": "Video processing has been queued.", "video_id": request.video_id}

@app.get("/health")
def health_check():
    """A simple health check endpoint to confirm the service is running."""
    return {"status": "ok"}