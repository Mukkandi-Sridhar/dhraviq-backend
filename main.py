# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
import traceback

# Load environment variables
load_dotenv()

# Firebase (safe initialization)
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
except Exception as e:
    print(f"🔥 Firebase initialization failed:\n{traceback.format_exc()}")
    db = None

# ------------------ Request Schema ------------------
class RunAgentRequest(BaseModel):
    user_id: str
    goals: Optional[List[str]] = None
    email: Optional[str] = None

# ------------------ Simulated Agent Runner ------------------
def agentic_workflow(user_id: str, goals: Optional[List[str]]):
    # Simulated agents
    agents = [
        {"agent": "GoalClarifier", "response": f"Clarified goals for {user_id}"},
        {"agent": "TimelineWizard", "response": "Suggested roadmap with milestones."}
    ]

    # Attempt to store session in Firestore
    if db:
        try:
            doc_ref = db.collection("sessions").document()
            doc_ref.set({
                "userId": user_id,
                "goals": goals or ["Default goal"],
                "agents": [a["agent"] for a in agents],
                "responses": {a["agent"]: a["response"] for a in agents},
                "createdAt": datetime.utcnow()
            })
            print("✅ Session stored in Firestore.")
        except Exception as e:
            print(f"⚠️ Error writing to Firestore:\n{traceback.format_exc()}")
    else:
        print("⚠️ Firebase is not connected. Skipping session save.")

    return {
        "user_id": user_id,
        "goals": goals or ["Default goal"],
        "results": agents,
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
        "firebase": "connected" if db else "not connected",
        "message": "Dhraviq backend is live 🔥"
    }

# ------------------ Core Run Agents Endpoint ------------------
@app.post("/run_agents", tags=["Core Agents"])
async def run_agents(data: RunAgentRequest):
    try:
        result = agentic_workflow(user_id=data.user_id, goals=data.goals)
        return result
    except Exception as e:
        print(f"❌ Unexpected error in /run_agents:\n{traceback.format_exc()}")
        raise HTTPException(500, detail="Something went wrong while running agents.")
