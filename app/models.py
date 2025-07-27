from pydantic import BaseModel, HttpUrl

# Pydantic model to validate the incoming request body
class VideoRequest(BaseModel):
    video_url: HttpUrl
    video_id: str