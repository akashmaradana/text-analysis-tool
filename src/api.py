from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from .summarizer.abstractive import BartSummarizer
from .config import BART_MODEL_NAME, MAX_INPUT_LENGTH, MIN_INPUT_LENGTH

app = FastAPI()

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 130
    min_length: Optional[int] = 30

class SummarizationResponse(BaseModel):
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float

# Initialize the summarizer
summarizer = BartSummarizer(BART_MODEL_NAME)

@app.get("/health")
async def health_check():
    """Check if the API and model are healthy"""
    return {
        "status": "healthy",
        "model": "loaded",
        "device": summarizer.device
    }

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """Generate a summary for the given text"""
    # Validate input length
    if len(request.text) < MIN_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input text must be at least {MIN_INPUT_LENGTH} characters long"
        )
    if len(request.text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Input text must not exceed {MAX_INPUT_LENGTH} characters"
        )

    try:
        # Update summarizer config if needed
        summarizer.update_config({
            "max_length": request.max_length,
            "min_length": request.min_length
        })

        # Generate summary
        summary = summarizer.summarize(request.text)

        # Calculate statistics
        original_length = len(request.text.split())
        summary_length = len(summary.split())
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        return SummarizationResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 