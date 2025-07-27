import os
import yt_dlp
from fastembed import TextEmbedding
from pinecone import Pinecone
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
print(api_key)
pinecone_index = pc.Index("superllm")

# Initialize the embedding model from fast-embed
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Initialize the transcription model (ONNX-based, fast and light)
transcription_model = WhisperModel("base", device="cpu")  # or "cuda" if GPU available

def process_video_and_upsert(video_url: str, video_id: str):
    """
    Orchestrates downloading, transcribing, embedding, and storing video content.
    """
    try:
        # Step 1: Download audio from the YouTube URL
        audio_file = download_audio(video_url, video_id)

        # Step 2: Transcribe the audio to text using faster-whisper
        segments, _ = transcription_model.transcribe(audio_file)
        transcript = " ".join([segment.text for segment in segments])
        os.remove(audio_file)  # Clean up the audio file immediately

        # Step 3: Split the transcript into smaller, manageable chunks
        chunks = chunk_text(transcript)

        # Step 4: Generate embeddings for each text chunk
        embeddings = list(embedding_model.embed(chunks))

        # Step 5: Prepare vectors for Pinecone upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                "id": f"{video_id}-{i}",
                "values": embedding,
                "metadata": {"text": chunk, "video_id": video_id}
            }
            vectors_to_upsert.append(vector)

        # Step 6: Upsert the vectors into the Pinecone index
        pinecone_index.upsert(vectors=vectors_to_upsert)

        print(f"Successfully processed and upserted video: {video_id}")
        return {"status": "success", "message": f"Video {video_id} processed."}

    except Exception as e:
        print(f"Error processing {video_id}: {e}")
        return {"status": "error", "message": str(e)}

def download_audio(video_url: str, video_id: str) -> str:
    """Downloads audio from a YouTube URL to a temporary file."""
    output_template = f"./{video_id}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'outtmpl': output_template,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(video_url, download=True)
    return f"{video_id}.mp3"

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits a long text into smaller, overlapping chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
