import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")

client = genai.Client(
    api_key=API_KEY,
    vertexai=False
)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Explain RAG in simple words."
    )

    print(response.text)

except Exception as error:
    print("Error:", error)

finally:
    client.close()