import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from mali.generate import load_mali_model, generate_text

# Global state for loaded model and tokenizer
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path = os.getenv("MALI_CHECKPOINT", "checkpoints/mali_best.pt")
    vocab_path = os.getenv("MALI_VOCAB", "checkpoints/mali_vocab.json")
    
    print(f"[API] Loading Mali Transformer from {checkpoint_path}...")
    try:
        model, tokenizer, device = load_mali_model(checkpoint_path, vocab_path)
        state["model"] = model
        state["tokenizer"] = tokenizer
        state["device"] = device
        state["num_params"] = model.get_num_params()
        print(f"[API] ✅ Model loaded successfully on {device.upper()} with {state['num_params']:,} parameters.")
    except Exception as e:
        print(f"[API Warning] Model not loaded at startup ({e}). Train the model first or pass valid checkpoint.")
    yield
    state.clear()

app = FastAPI(
    title="Mali Language Model API",
    description="Production REST API serving the custom Mali Transformer architecture.",
    version="1.0.0",
    lifespan=lifespan
)

class GenerateRequest(BaseModel):
    prompt: str = Field(default="To be, or not to be,", description="Prompt text")
    max_tokens: int = Field(default=150, ge=1, le=500, description="Tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Creativity temperature")
    top_k: int = Field(default=40, ge=1, le=100, description="Top-K cutoff")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Nucleus sampling threshold")

class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float

@app.get("/health")
def health_check():
    if "model" not in state:
        return {"status": "degraded", "message": "Model weights not loaded yet"}
    return {
        "status": "healthy",
        "model": "Mali-Transformer",
        "parameters": state["num_params"],
        "device": state["device"]
    }

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure checkpoints exist.")
    
    t0 = time.time()
    generated = generate_text(
        state["model"],
        state["tokenizer"],
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        device=state["device"]
    )
    latency_ms = (time.time() - t0) * 1000.0
    tokens_per_sec = (req.max_tokens / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

    return GenerateResponse(
        prompt=req.prompt,
        completion=generated,
        tokens_generated=req.max_tokens,
        latency_ms=round(latency_ms, 2),
        tokens_per_second=round(tokens_per_sec, 2)
    )
