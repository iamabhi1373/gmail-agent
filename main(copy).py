import os
import base64
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

load_dotenv()

class GmailAIAgent:
    def __init__(self, credentials_file, gemini_api_key=None):
        self.SCOPES = ['https://www.googleapis.com/auth/gmail.send']
        self.credentials_file = credentials_file
        self.creds = None
        self.service = None

        api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        self.authenticate()

    def authenticate(self):
        token_file = 'token.json'

        if os.path.exists(token_file):
            self.creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    if os.path.exists(token_file):
                        os.remove(token_file)
                    self.creds = None

            if not self.creds:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                self.creds = flow.run_local_server(port=0)

            with open(token_file, 'w') as token:
                token.write(self.creds.to_json())
            os.chmod(token_file, 0o600)

        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
        except Exception as e:
            raise Exception(f"Failed to build Gmail service: {e}")

    def validate_email(self, email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def generate_email_content(self, prompt, context="", recipient_type="professional", sender_name="", relationship_details=""):
        base_instructions = """You are a helpful email writing assistant. Write a natural, human-like email based on the given task. 
        Format your response exactly as follows:
        
        Subject: [Your email subject here]
        Body: [Your email body here]
        
        Make the email sound natural and conversational, not robotic or AI-generated."""

        if recipient_type == "loved_one":
            full_prompt = f"""{base_instructions}
            
            Context: {context}
            Sender: {sender_name}
            Relationship: {relationship_details}
            Task: {prompt}
            
            Write a warm, affectionate email to a loved one."""
        elif recipient_type == "friend":
            full_prompt = f"""{base_instructions}
            
            Context: {context}
            Sender: {sender_name}
            Relationship: {relationship_details}
            Task: {prompt}
            
            Write a casual, friendly email to a friend."""
        elif recipient_type == "family":
            full_prompt = f"""{base_instructions}
            
            Context: {context}
            Sender: {sender_name}
            Relationship: {relationship_details}
            Task: {prompt}
            
            Write a warm, family-oriented email."""
        else:
            full_prompt = f"""{base_instructions}
            
            Context: {context}
            Sender: {sender_name}
            Relationship: {relationship_details}
            Task: {prompt}
            
            Write a professional email."""

        try:
            response = self.model.generate_content(full_prompt)
            if not response or not response.text:
                print("Empty response from Gemini API")
                return None, None

            subject, body = self.parse_email_response(response.text)
            if subject and body:
                subject, body = self._refine_generated_content(subject, body, recipient_type)

            return subject, body

        except Exception as e:
            print(f"Error generating content: {e}")
            return None, None

    def _refine_generated_content(self, subject, body, recipient_type):
        ai_phrases = {
            "I hope this email finds you well": ["Hope you're doing well", "Hope things are going well", ""],
            # ... other phrases
        }

        for formal, casual_options in ai_phrases.items():
            if formal.lower() in body.lower():
                import random
                replacement = random.choice(casual_options)
                body = body.replace(formal, replacement)

        body = re.sub(r'!{2,}', '!', body)
        excessive_patterns = [
            (r'\b(absolutely|extremely|incredibly|truly|really|very)\s+(amazing|wonderful|fantastic|incredible|perfect)\b',
             lambda m: m.group(2)),
            (r'\b(so so|very very)\b', 'so'),
        ]
        for pattern, replacement in excessive_patterns:
            body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)

        generic_subjects = ["Hello", "Hi there", "Greetings", "Good day"]
        if subject.strip() in generic_subjects:
            if recipient_type == "loved_one":
                subject = "Hey you ❤️"
            elif recipient_type == "friend":
                subject = "Hey!"
            else:
                subject = "Quick note"

        return subject.strip(), body.strip()

    def parse_email_response(self, response_text):
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

        # If parsing failed, try to extract content more flexibly
        if not subject or not body:
            print(f"Standard parsing failed - Subject: {'✓' if subject else '✗'}, Body: {'✓' if body else '✗'}")
            print("Attempting flexible parsing...")
            
            # Try to find subject and body in the response
            text = response_text.strip()
            
            # Look for subject patterns
            subject_patterns = [
                r'Subject:\s*(.+)',
                r'Subject\s*:\s*(.+)',
                r'^(.+?)(?:\n|$)'
            ]
            
            for pattern in subject_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and not subject:
                    subject = match.group(1).strip()
                    break
            
            # Look for body patterns
            body_patterns = [
                r'Body:\s*(.+)',
                r'Body\s*:\s*(.+)',
                r'(?:Subject:.*?\n)(.+)',
                r'^.*?\n(.+)'
            ]
            
            for pattern in body_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match and not body:
                    body = match.group(1).strip()
                    break
            
            # If still no body, use everything after the first line as body
            if not body and subject:
                lines = text.split('\n')
                if len(lines) > 1:
                    body = '\n'.join(lines[1:]).strip()

        if not subject or not body:
            print(f"Failed to parse email - Subject: {'✓' if subject else '✗'}, Body: {'✓' if body else '✗'}")
            print(f"Raw response: {response_text[:200]}...")
            return None, None

        return subject, body.strip()

    def create_message(self, to_email, subject, body, from_email):
        if not self.validate_email(to_email):
            raise ValueError(f"Invalid recipient email: {to_email}")

        message = MIMEMultipart()
        message['to'] = to_email
        message['from'] = from_email
        message['subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return {'raw': raw_message}

    def send_email(self, to_email, prompt, context="", from_email="me", recipient_type="professional", sender_name="", relationship_details=""):
        print(f"Generating {recipient_type} email for: {to_email}")

        subject, body = self.generate_email_content(
            prompt=prompt,
            context=context,
            recipient_type=recipient_type,
            sender_name=sender_name,
            relationship_details=relationship_details
        )

        if not subject or not body:
            print("Failed to generate email content")
            return False

        print(f"Generated email:\nSubject: {subject}\nBody preview: {body[:150]}...")

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
        


if __name__ == "__main__":
    try:
        agent = GmailAIAgent(credentials_file='credentials.json')

        success = agent.send_email(
            to_email="raisiddharth158@gmail.com",
            prompt="write a email to my friend saying hi and tell him that her love Manya aagi is in the campus wearing ethical cloth looking very hot and sexy.",
            context="naugthy email teasing him.",
            recipient_type="friend",
            sender_name="ABHSIEHK",
            relationship_details="my old friend"
        )

        if success:
            print("Email sent successfully!")
        else:
            print("Failed to send email.")

    except Exception as e:
        print(f"Error: {e}")

