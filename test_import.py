import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY", "").strip()

if not key:
    raise ValueError("Missing GEMINI_API_KEY")

print("API key loaded:", bool(key))
print("Key length:", len(key))
print("Key prefix:", key[:3])

client = genai.Client(
    api_key=key,
    vertexai=False
)

try:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents="This is a test document.",
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
    )

    embedding = response.embeddings[0].values

    print("SUCCESS!")
    print("Embedding dimensions:", len(embedding))

except Exception as error:
    print("FAILED:")
    print(error)

finally:
    client.close()