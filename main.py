import os
import base64
import json
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GmailAIAgent:
    def __init__(self, credentials_file, gemini_api_key=None):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send']
        self.credentials_file = credentials_file
        self.creds = None
        self.service = None
        
        # Configure Gemini - Use environment variable if not provided
        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        token_file = 'token.json'
        
        if os.path.exists(token_file):
            self.creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    # Remove invalid token file
                    if os.path.exists(token_file):
                        os.remove(token_file)
                    self.creds = None
            
            if not self.creds:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                self.creds = flow.run_local_server(port=0)
            
            # Save token with restricted permissions
            with open(token_file, 'w') as token:
                token.write(self.creds.to_json())
            os.chmod(token_file, 0o600)  # Read/write for owner only
        
        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
        except Exception as e:
            raise Exception(f"Failed to build Gmail service: {e}")
    
    def validate_email(self, email):
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def generate_email_content(self, prompt, context="", recipient_type="professional"):
        """
        Generate email content using Gemini API.
        recipient_type: 'professional' or 'loved_one'
        """
        if recipient_type == "loved_one":
            full_prompt = f"""
            Context: {context}

            Task: {prompt}

            Please generate a warm, heartfelt email for a loved one with:
            - A friendly, affectionate subject line
            - A caring, personal body
            - An informal, loving tone

            Format your response EXACTLY as follows:
            Subject: 

            Body:
            [your email body here]
            """
        else:
            full_prompt = f"""
            Context: {context}

            Task: {prompt}

            Please generate a professional email with:
            - Appropriate subject line
            - Well-structured body
            - Professional tone

            Format your response EXACTLY as follows:
            Subject: 

            Body:
            [your email body here]
            """
        try:
            response = self.model.generate_content(full_prompt)
            if not response or not response.text:
                print("Empty response from Gemini API")
                return None, None
            return self.parse_email_response(response.text)
        except Exception as e:
            print(f"Error generating content: {e}")
            return None, None

    def parse_email_response(self, response_text):
        """Parse Gemini response to extract subject and body"""
        if not response_text:
            return None, None
        lines = response_text.strip().split('\n')
        subject = ""
        body = ""
        body_started = False
        for line in lines:
            line = line.strip()
            if line.startswith("Subject:") and not subject:
                subject = line.replace("Subject:", "").strip()
            elif line.startswith("Body:"):
                body_started = True
                body_content = line.replace("Body:", "").strip()
                if body_content:
                    body = body_content
            elif body_started:
                if body:
                    body += "\n" + line
                else:
                    body = line
        if not subject or not body:
            print(f"Failed to parse email - Subject: {'✓' if subject else '✗'}, Body: {'✓' if body else '✗'}")
            return None, None
        return subject, body.strip()

    def create_message(self, to_email, subject, body, from_email):
        """Create email message"""
        if not self.validate_email(to_email):
            raise ValueError(f"Invalid recipient email: {to_email}")
        message = MIMEMultipart()
        message['to'] = to_email
        message['from'] = from_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return {'raw': raw_message}

    def send_email(self, to_email, prompt, context="", from_email="me"):
        """Generate and send email"""
        print(f"Generating email for: {to_email}")
        subject, body = self.generate_email_content(prompt, context)
        if not subject or not body:
            print("Failed to generate email content")
            return False
        print(f"Generated email:")
        print(f"Subject: {subject}")
        print(f"Body preview: {body[:100]}...")
        try:
            message = self.create_message(to_email, subject, body, from_email)
            result = self.service.users().messages().send(userId='me', body=message).execute()
            print(f"Email sent successfully! Message ID: {result['id']}")
            return True
        except HttpError as e:
            print(f"Gmail API error: {e}")
            return False
        except Exception as e:
            print(f"Error sending email: {e}")
            return False

# Usage Example
if __name__ == "__main__":
    try:
        # Initialize the agent - API key should be in environment variable
        agent = GmailAIAgent(
            credentials_file='credentials.json'
            # gemini_api_key will be read from GEMINI_API_KEY environment variable
        )
        
        # Send an email
        success = agent.send_email(
            to_email="sachan.addya@gmail.com",
            prompt="write a email to my loved ones saying sorry and cute message telling her she is the most beautiful and amazing girl i met.",
            context="write a email to my loved ones"
        )
        
        if success:
            print("Email sent successfully!")
        else:
            print("Failed to send email.")
            
    except Exception as e:
        print(f"Error: {e}")

import google.generativeai as genai
import os

print("\n--- Gemini: Listing available models for your API key ---")
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("GEMINI_API_KEY environment variable is not set. Please set it and rerun.")
    exit(1)
genai.configure(api_key=api_key)
try:
    models = genai.list_models()
    print("Available models:")
    for m in models:
        print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
    exit(1)
print("--- End of model list ---\n")