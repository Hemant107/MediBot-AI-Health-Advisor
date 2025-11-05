import base64
import io
import os
import logging
import requests
from PIL import Image
from dotenv import load_dotenv

# ---------------------- Setup ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY is not set in the .env file")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------- Models ----------------------
TEXT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ---------------------- API Call ----------------------
def call_groq_api(model, messages):
    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.3
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Groq API Error: {response.status_code} - {response.text}")
            return {"error": f"API Error: {response.status_code} - {response.text}"}

        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"response": answer}

    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return {"error": str(e)}

# ---------------------- Text-only Query ----------------------
def process_text(query):
    messages = [{"role": "user", "content": query}]
    return call_groq_api(TEXT_MODEL, messages)

# ---------------------- Vision Query ----------------------
def process_image(image_path, query):
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((1024, 1024))

        encoded_image = base64.b64encode(img.tobytes()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        return call_groq_api(VISION_MODEL, messages)

    except Exception as e:
        logger.error(f"Unexpected Error: {str(e)}")
        return {"error": str(e)}


# ---------------------- Example ----------------------
if __name__ == "__main__":
    # Text example
    text_query = "Explain the symptoms of diabetes."
    print("Text-only model response:")
    text_result = process_text(text_query)
    print(text_result.get("response", text_result.get("error")))

    # Vision example
    image_query = "Describe the objects and text in this image."
    image_path = "test1.png"  # Replace with your image path
    print("\nVision model response:")
    vision_result = process_image(image_path, image_query)
    print(vision_result.get("response", vision_result.get("error")))
