# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from datetime import datetime
import traceback

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
except Exception as e:
    print(f"🔥 Firebase initialization failed:\n{traceback.format_exc()}")
    db = None

# ------------------ Request Schema ------------------
class RunAgentRequest(BaseModel):
    user_id: str
    message: str
    email: Optional[str] = None

# ------------------ Simulated Agent Handler ------------------
def agentic_workflow(user_id: str, message: str):
    # Simulate an agent response
    response_text = f"Hi {user_id}, you said: '{message}'"

    # Try saving to Firestore
    if db:
        try:
            db.collection("sessions").document().set({
                "userId": user_id,
                "message": message,
                "response": response_text,
                "createdAt": datetime.utcnow()
            })
            print("✅ Message stored in Firestore.")
        except Exception as e:
            print(f"⚠️ Firestore write failed:\n{traceback.format_exc()}")
    else:
        print("⚠️ Firebase not connected. Skipping Firestore save.")

    return {
        "user_id": user_id,
        "message": message,
        "response": response_text,
        "status": "success"
    }

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Accepts chat messages, responds, and stores them if Firebase is available.",
    version="2.0.0"
)

# ------------------ CORS Settings ------------------
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

# ------------------ POST /run_agents Endpoint ------------------
@app.post("/run_agents", tags=["Core Agents"])
async def run_agents(data: RunAgentRequest):
    try:
        result = agentic_workflow(user_id=data.user_id, message=data.message)
        return result
    except Exception as e:
        print(f"❌ Error in /run_agents:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error. Please try again.")
