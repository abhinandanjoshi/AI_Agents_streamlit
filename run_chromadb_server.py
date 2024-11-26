from chromadb import Client
from chromadb.config import Settings

# Initialize ChromaDB server settings
client = Client(Settings(
    chroma_api_impl="chromadb.api.fastapi.FastAPI",  # Use FastAPI for HTTP
    chroma_server_host="localhost",                    # Host IP (allows connections from all IPs)
    chroma_server_http_port=8000                     # Port (default: 8000)
))

print("ChromaDB server is running on http://0.0.0.0:8000")
