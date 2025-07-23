from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------ Firebase Initialization ------------------
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

# ------------------ Request Schema ------------------
class RunAgentRequest(BaseModel):
    userId: str
    question: str
    agents: List[str]
    email: Optional[str] = None
    send_email: Optional[bool] = False

# ------------------ Simulated Agent Handler ------------------
def agentic_workflow(user_id: str, question: str, agents: List[str]) -> dict:
    responses = {}

    for agent in agents:
        responses[agent] = f"✅ {agent} received your query: '{question}'"

    # Firestore write
    if db:
        try:
            db.collection("sessions").add({
                "userId": user_id,
                "question": question,
                "agents": agents,
                "responses": responses,
                "email": None,
                "send_email": False,
                "createdAt": datetime.utcnow()
            })
            print("✅ Firestore logged the session.")
        except Exception:
            print(f"⚠️ Firestore logging failed:\n{traceback.format_exc()}")

    return {"responses": responses}

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Handles multi-agent chat and logs sessions",
    version="2.1.0"
)

# ------------------ CORS Middleware ------------------
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
        "firebase": "connected" if db else "not connected",
        "message": "Dhraviq backend is live 🔥"
    }

# ------------------ Core Endpoint ------------------
@app.post("/run_agents", tags=["Core Agents"])
async def run_agents(data: RunAgentRequest, authorization: Optional[str] = Header(None)):
    try:
        print(f"📩 Incoming request from: {data.userId} | Agents: {data.agents}")
        result = agentic_workflow(data.userId, data.question, data.agents)
        return result
    except Exception:
        print(f"❌ Exception in /run_agents:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")
