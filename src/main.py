from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .api import app as api_app

app = FastAPI(title="Text Summarization Service")

# Mount the API under /api
app.mount("/api", api_app)

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}

@app.post("/summarize")
async def summarize_text(text: str, max_length: int = 130, min_length: int = 30):
    if len(text) < MIN_INPUT_LENGTH:
        return {"error": f"Text must be at least {MIN_INPUT_LENGTH} characters long"}
    if len(text) > MAX_INPUT_LENGTH:
        return {"error": f"Text must not exceed {MAX_INPUT_LENGTH} characters"}

    try:
        # Update configuration if needed
        if max_length != 130 or min_length != 30:
            summarizer.update_config({
                "max_length": max_length,
                "min_length": min_length
            })

        # Generate summary
        summary = summarizer.summarize(text)

        # Calculate statistics
        original_length = len(text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        return {
            "summary": summary,
            "original_length": original_length,
            "summary_length": summary_length,
            "compression_ratio": compression_ratio
        }
    except Exception as e:
        return {"error": str(e)} 