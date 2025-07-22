from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentic_ai_backend import run_agents, send_pushover_notification
import os
from dotenv import load_dotenv

# ------------------ Load Environment ------------------
load_dotenv()

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Multi-agent orchestration using Gemini, Firestore & Pushover",
    version="1.0.0"
)

# ------------------ CORS Middleware ------------------
# Allow both production and local development domains
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
