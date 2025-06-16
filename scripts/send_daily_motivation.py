import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

# Firebase initialization
firebase_path = os.getenv("FIREBASE_CRED_PATH", "firebase_credentials.json")
if not os.path.exists(firebase_path):
    raise FileNotFoundError(f"Missing Firebase credential file at: {firebase_path}")

cred = credentials.Certificate(firebase_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# SendGrid email sender
def send_email(to_email, subject, body):
    try:
        message = Mail(
            from_email="no-reply@dhraviq.com",
            to_emails=to_email,
            subject=subject,
            html_content=f"""
                <html>
                    <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f9fafb;">
                        <div style="max-width: 600px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                            <h2 style="color: #4f46e5;">🌟 Your Daily Motivation</h2>
                            <p style="font-size: 16px; color: #374151;">{body.replace('\n', '<br>')}</p>
                            <hr style="margin: 24px 0;">
                            <p style="font-size: 13px; color: #6b7280;">Stay consistent. You're building something incredible!</p>
                            <p style="font-size: 13px; color: #9ca3af;">— Team Dhraviq</p>
                        </div>
                    </body>
                </html>
            """
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# Extract the day's section from plan
def extract_day_content(plan, day):
    start = plan.find(f"**Day {day}:")
    end = plan.find(f"**Day {day + 1}:")
    if start == -1:
        return None
    return plan[start:end].strip() if end != -1 else plan[start:].strip()

# Main loop
def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    today_index = (datetime.utcnow().day % 7) or 7  # Day 1–7 rotation

    users = db.collection("users").where("reminderEnabled", "==", True).stream()

    for user_doc in users:
        user = user_doc.to_dict()
        email = user.get("email")
        plan = user.get("reminderPlan")
        last_sent = user.get("lastEmailSent")

        if not email or not plan:
            continue
        if last_sent == today_str:
            print(f"⏭ Already sent to {email} today")
            continue

        content = extract_day_content(plan, today_index)
        if content:
            send_email(email, f"🌅 Day {today_index} - Your Dhraviq Goal Boost", content)
            db.collection("users").document(user_doc.id).update({
                "lastEmailSent": today_str
            })

if __name__ == "__main__":
    main()
