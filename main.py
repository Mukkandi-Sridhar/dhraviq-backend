# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------ Define Request Schema ------------------
class RunAgentRequest(BaseModel):
    user_id: str
    goals: Optional[List[str]] = None
    email: Optional[str] = None

# Simulated agent runner
def agentic_workflow(user_id: str, goals: Optional[List[str]]):
    return {
        "user_id": user_id,
        "goals": goals or ["Default goal"],
        "status": "success",
        "message": "Agents ran successfully!"
    }

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Multi-agent orchestration using Gemini, Firestore & Pushover",
    version="1.0.0"
)

# ------------------ CORS ------------------
ALLOWED_ORIGINS = [
    "https://dhraviq.com",
    "https://www.dhraviq.com",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Health Check ------------------
@app.get("/health", include_in_schema=False)
@app.head("/health")
def health_check():
    return {
        "status": "OK",
        "message": "Dhraviq backend is live 🔥"
    }

# ------------------ Core Run Agents Endpoint ------------------
@app.post("/run_agents", tags=["Core Agents"])
async def run_agents(data: RunAgentRequest):
    result = agentic_workflow(user_id=data.user_id, goals=data.goals)
    return result
