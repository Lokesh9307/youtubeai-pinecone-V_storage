import os
from fastembed import TextEmbedding
from pinecone import Pinecone
from faster_whisper import WhisperModel
from pytube import YouTube
from dotenv import load_dotenv
import traceback

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
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

        print(f"✅ Successfully processed and upserted video: {video_id}")
        return {"status": "success", "message": f"Video {video_id} processed."}

    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def download_audio(video_url: str, video_id: str) -> str:
    try:
        print(f"Starting YouTube download: {video_url}")
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise Exception("No audio stream found.")
        output_file = f"{video_id}.mp4"
        stream.download(filename=output_file)
        print(f"Downloaded audio to {output_file}")
        return output_file
    except Exception as e:
        print(f"Download error: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits a long text into smaller, overlapping chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
