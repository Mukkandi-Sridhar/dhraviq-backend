from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import traceback
import os
from dotenv import load_dotenv
from agentic_ai_backend import run_agentic_logic  # External logic

load_dotenv()

# Firebase Setup
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    cred_path = "firebase_credentials.json"
    if os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized successfully.")
    else:
        print(f"❌ Firebase credential file not found at: {cred_path}")
        db = None
except Exception:
    print(f"🔥 Firebase initialization failed:\n{traceback.format_exc()}")
    db = None

# Request Schema
class RunAgentRequest(BaseModel):
    userId: str
    question: str
    agents: List[str]
    email: Optional[str] = None
    send_email: Optional[bool] = False

# FastAPI Setup
app = FastAPI(
    title="Dhraviq Agentic AI Gateway",
    description="Routes requests to agent logic and logs sessions",
    version="2.2.0"
)

# CORS
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

@app.get("/health", include_in_schema=False)
@app.head("/health")
def health_check():
    return {
        "status": "OK",
        "firebase": "connected" if db else "not connected",
        "message": "Dhraviq gateway is live 🔥"
    }

@app.post("/run_agents", tags=["Core Agents"])
async def run_agents(data: RunAgentRequest, authorization: Optional[str] = Header(None)):
    try:
        print(f"📩 Incoming request from: {data.userId} | Agents: {data.agents}")
        result = await run_agentic_logic(data)
        return result
    except Exception:
        print(f"❌ Exception in /run_agents:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")
