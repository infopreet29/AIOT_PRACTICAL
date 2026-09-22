from fastapi import FastAPI
from pydantic import BaseModel

# Create the FastAPI app instance
app = FastAPI(title="My First AI API")

# A simple GET endpoint
@app.get("/")
def home():
    return {"message": "Welcome to my AI API"}

# Define the structure of incoming data using Pydantic
class TextRequest(BaseModel):
    text: str

# A POST endpoint that "processes" text (simulating an AI task)
@app.post("/analyze")
def analyze_text(request: TextRequest):
    word_count = len(request.text.split())
    return {
        "original_text": request.text,
        "word_count": word_count,
        "message": "Text analyzed successfully"
    }
