from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import base64, io, logging
from PIL import Image
from main import process_text, process_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Ensure a static folder exists
import os
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    try:
        # Read uploaded image
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        # Save temporarily to process
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Process both models
        llama_response = process_text(query).get("response", "No response")
        llava_response = process_image(temp_path, query).get("response", "No response")

        # Remove temporary file
        import os
        os.remove(temp_path)

        return JSONResponse(content={"llama": llama_response, "llava": llava_response})

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")
