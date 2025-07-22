from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentic_ai_backend import run_agents, send_pushover_notification

import os
from dotenv import load_dotenv

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials

# ------------------ Load Environment ------------------
load_dotenv()

# ------------------ Initialize Firebase ------------------
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH", "dhraviq-firebase-adminsdk-fbsvc-00ee4536d0.json")

# Only initialize once to avoid errors during reloads
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred)

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Multi-agent orchestration using Gemini, Firestore & Pushover",
    version="1.0.0"
)

# ------------------ CORS Middleware ------------------
ALLOWED_ORIGINS = [
    "https://dhraviq.com",
    "https://www.dhraviq.com",
    "http://localhost:5173",
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

# ------------------ Pushover Notification Test ------------------
@app.get("/test_notification", tags=["Diagnostics"])
def test_notification():
    success = send_pushover_notification(
        user_id="test-user",
        question="This is a test notification from /test_notification",
        email="test@example.com"
    )
    return {
        "sent": success,
        "message": (
            "✅ If 'sent' is true, you should receive a test notification "
            "on your Pushover mobile app."
        )
    }

# ------------------ Run Agents Endpoint ------------------
app.post("/run_agents", tags=["Core Agents"])(run_agents)
