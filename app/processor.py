import os
import traceback
import numpy as np
from fastembed import TextEmbedding
from pinecone import Pinecone
from faster_whisper import WhisperModel
from pytube import YouTube
from dotenv import load_dotenv

load_dotenv()

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("Missing Pinecone API key in environment variables.")

pc = Pinecone(api_key=api_key)
pinecone_index = pc.Index("superllm")  # Ensure the index "superllm" exists

# Embedding model
embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
expected_dim = 384  # Dimension of all-MiniLM-L6-v2

# Transcription model
transcription_model = WhisperModel("base", device="cpu")

def process_video_and_upsert(video_url: str, video_id: str):
    try:
        print(f"\n🎬 Processing video ID: {video_id}")

        # Step 1: Download audio
        audio_file = download_audio(video_url, video_id)

        # Step 2: Transcribe
        print("🔊 Transcribing...")
        segments, _ = transcription_model.transcribe(audio_file)
        transcript = " ".join([segment.text for segment in segments])
        os.remove(audio_file)

        # Step 3: Chunk text
        chunks = chunk_text(transcript)
        print(f"📄 Transcript split into {len(chunks)} chunks.")

        # Step 4: Embed
        embeddings = list(embedding_model.embed(chunks))
        print("🔗 Embeddings generated.")

        # Step 5: Validate and prepare vectors
        vectors_to_upsert = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb = np.array(emb, dtype=np.float32).tolist()
            if len(emb) != expected_dim:
                raise ValueError(f"Invalid embedding length for chunk {i}: {len(emb)}")

            vectors_to_upsert.append({
                "id": f"{video_id}-{i}",
                "values": emb,
                "metadata": {
                    "text": chunk,
                    "video_id": video_id
                }
            })

        # Step 6: Upsert
        print(f"⬆️ Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        response = pinecone_index.upsert(vectors=vectors_to_upsert)
        print("✅ Upsert complete:", response)

        return {"status": "success", "message": f"Video {video_id} processed and stored."}

    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


def download_audio(video_url: str, video_id: str) -> str:
    try:
        print(f"⬇️ Downloading audio from: {video_url}")
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise Exception("No audio stream found.")
        filename = f"{video_id}.mp4"
        stream.download(filename=filename)
        print(f"📁 Audio saved as: {filename}")
        return filename
    except Exception as e:
        raise Exception(f"YouTube download failed: {e}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
