from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from agentic_ai_backend import run_agents, send_pushover_notification
import os
from dotenv import load_dotenv

# ------------------ Load Environment ------------------
load_dotenv()

# Optional: Print to verify .env is loaded correctly
print("PUSHOVER_TOKEN:", os.getenv("PUSHOVER_TOKEN"))
print("PUSHOVER_USER:", os.getenv("PUSHOVER_USER"))

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Multi-agent system using Gemini + Firebase",
    version="1.0.0"
)

# ------------------ CORS (adjust in prod) ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔐 Replace with ["https://dhraviq.com"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Health Check ------------------
@app.get("/health", include_in_schema=False)
@app.head("/health")
def health_check():
    return {"status": "OK", "message": "Dhraviq backend


# ------------------ Pushover Test Endpoint ------------------
@app.get("/test_notification")
def test_notification():
    success = send_pushover_notification(
        user_id="test-user",
        question="This is a test notification from /test_notification",
        email="test@example.com"
    )
    return {
        "sent": success,
        "message": "✅ If 'sent' is true, you should get this on your Pushover mobile app."
    }

# ------------------ POST /run_agents Endpoint ------------------
app.post("/run_agents")(run_agents)
