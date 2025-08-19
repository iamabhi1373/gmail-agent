import os
import base64
import json
import re
import pyttsx3
import threading
import time
import queue
import sounddevice as sd
import vosk
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

VOSK_MODEL_PATH = "/Users/macbook/Downloads/vosk-model-en-us-0.42-gigaspeech"

class VoiceGmailAIAgent:
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
        
        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            voices = self.engine.getProperty('voices')
            if voices:
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                else:
                    self.engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"Text-to-speech not available: {e}")
            self.engine = None
        
        # Initialize Vosk model for offline speech recognition
        try:
            self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
            self.vosk_samplerate = 16000
            self.vosk_queue = queue.Queue()
        except Exception as e:
            print(f"Vosk model not available: {e}")
            self.vosk_model = None
        
        self.authenticate()
    
    def speak(self, text):
        print(f"🤖 AI: {text}")
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
    
    def get_text_input(self, prompt):
        print(f"👤 {prompt}")
        return input("> ").strip()
    
    def listen_vosk(self, prompt=None, timeout=10):
        if not self.vosk_model:
            print("Vosk model not loaded.")
            return None
        if prompt:
            self.speak(prompt)
        print("🎤 Listening (Vosk)... Speak now.")
        q = self.vosk_queue
        rec = vosk.KaldiRecognizer(self.vosk_model, self.vosk_samplerate)
        def callback(indata, frames, time, status):
            q.put(bytes(indata))
        with sd.RawInputStream(samplerate=self.vosk_samplerate, blocksize=8000, dtype='int16', channels=1, callback=callback):
            result_text = ""
            start_time = time_module = time.time()
            while True:
                if not q.empty():
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        result = rec.Result()
                        text = json.loads(result).get("text", "")
                        if text:
                            print(f"👤 You said: {text}")
                            return text
                if time.time() - start_time > timeout:
                    print("⏰ Listening timed out.")
                    return None
    
    def voice_interaction(self):
        self.speak("Hello! I'm your voice-controlled Gmail AI assistant. How can I help you today?")
        print("Say 'compose email' to start, or 'quit' to exit.")
        while True:
            try:
                command = self.listen_vosk()
                if not command:
                    self.speak("I didn't catch that. Could you please repeat?")
                    continue
                command = command.lower()
                if any(word in command for word in ['quit', 'exit', 'stop', 'bye']):
                    self.speak("Goodbye! Have a great day!")
                    break
                elif any(word in command for word in ['compose', 'write', 'send', 'email', 'mail']):
                    self.compose_email_voice()
                elif any(word in command for word in ['help', 'what can you do']):
                    self.speak("I can help you compose and send emails using voice commands. Just say 'compose email' to start!")
                else:
                    response = self.generate_response(command)
                    if response:
                        self.speak(response)
                    else:
                        self.speak("I'm not sure how to help with that. Try saying 'compose email' or 'help' for assistance.")
            except KeyboardInterrupt:
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error in voice interaction: {e}")
                self.speak("Sorry, there was an error. Please try again.")
    
    def compose_email_voice(self):
        self.speak("I'll help you compose an email. Let's start with the recipient.")
        recipient = self.listen_vosk("Who would you like to send the email to? Please say the email address.")
        if not recipient:
            self.speak("I couldn't understand the email address. Let's try again.")
            return
        recipient = self.clean_email_address(recipient)
        subject_prompt = self.listen_vosk("What should the subject of the email be?")
        if not subject_prompt:
            self.speak("I couldn't understand the subject. Let's try again.")
            return
        content_prompt = self.listen_vosk("Now, please tell me what you'd like to say in the email. Take your time.")
        if not content_prompt:
            self.speak("I couldn't understand the email content. Let's try again.")
            return
        self.speak("Generating your email using AI...")
        full_prompt = f"""
        Create a professional email with:
        Subject: {subject_prompt}
        Content: {content_prompt}
        
        Format your response EXACTLY as follows:
        Subject: [your subject line here]
        
        Body:
        [your email body here]
        """
        try:
            response = self.model.generate_content(full_prompt)
            if response and response.text:
                subject, body = self.parse_email_response(response.text)
                if subject and body:
                    self.speak(f"I've generated an email with subject: {subject}")
                    self.speak("Would you like me to read the email content to you? Say yes or no.")
                    read_choice = self.listen_vosk()
                    if read_choice and any(word in read_choice for word in ['yes', 'read', 'sure']):
                        self.speak("Here's the email content:")
                        preview = body[:200] + "..." if len(body) > 200 else body
                        self.speak(preview)
                    self.speak("Would you like me to send this email? Say yes to send, or no to cancel.")
                    send_choice = self.listen_vosk()
                    if send_choice and any(word in send_choice for word in ['yes', 'send', 'sure', 'okay']):
                        success = self.send_email(recipient, subject, body)
                        if success:
                            self.speak("Great! Your email has been sent successfully!")
                        else:
                            self.speak("Sorry, there was an error sending the email. Please try again.")
                    else:
                        self.speak("Email cancelled. No worries!")
                else:
                    self.speak("Sorry, I couldn't generate a proper email. Let's try again.")
            else:
                self.speak("Sorry, I couldn't generate the email content. Let's try again.")
        except Exception as e:
            print(f"Error generating email: {e}")
            self.speak("Sorry, there was an error generating the email. Please try again.")
    
    def text_interaction(self):
        """Main text-based interaction loop"""
        self.speak("Hello! I'm your Gmail AI assistant. How can I help you today?")
        print("\nAvailable commands:")
        print("- 'compose email' - Create and send an email")
        print("- 'help' - Show available commands")
        print("- 'quit' - Exit the program")
        
        while True:
            try:
                # Get command
                command = self.get_text_input("What would you like to do?")
                if not command:
                    continue
                
                command = command.lower()
                
                # Process commands
                if any(word in command for word in ['quit', 'exit', 'stop', 'bye']):
                    self.speak("Goodbye! Have a great day!")
                    break
                
                elif any(word in command for word in ['compose', 'write', 'send', 'email', 'mail']):
                    self.compose_email_text()
                
                elif any(word in command for word in ['help', 'what can you do']):
                    self.speak("I can help you compose and send emails. Just say 'compose email' to start!")
                    print("\nAvailable commands:")
                    print("- 'compose email' - Create and send an email")
                    print("- 'help' - Show available commands")
                    print("- 'quit' - Exit the program")
                
                else:
                    # Try to generate a response using Gemini
                    response = self.generate_response(command)
                    if response:
                        self.speak(response)
                    else:
                        self.speak("I'm not sure how to help with that. Try saying 'compose email' or 'help' for assistance.")
                
            except KeyboardInterrupt:
                self.speak("Goodbye!")
                break
            except Exception as e:
                print(f"Error in interaction: {e}")
                self.speak("Sorry, there was an error. Please try again.")
    
    def compose_email_text(self):
        """Text-guided email composition"""
        self.speak("I'll help you compose an email. Let's start with the recipient.")
        
        # Get recipient
        recipient = self.get_text_input("Who would you like to send the email to? (email address)")
        if not recipient:
            self.speak("No email address provided. Let's try again.")
            return
        
        # Clean up email address
        recipient = self.clean_email_address(recipient)
        
        # Get subject
        subject_prompt = self.get_text_input("What should the subject of the email be?")
        if not subject_prompt:
            self.speak("No subject provided. Let's try again.")
            return
        
        # Get email content
        content_prompt = self.get_text_input("What would you like to say in the email?")
        if not content_prompt:
            self.speak("No content provided. Let's try again.")
            return
        
        # Generate email using AI
        self.speak("Generating your email using AI...")
        
        full_prompt = f"""
        Create a professional email with:
        Subject: {subject_prompt}
        Content: {content_prompt}
        
        Format your response EXACTLY as follows:
        Subject: [your subject line here]
        
        Body:
        [your email body here]
        """
        
        try:
            response = self.model.generate_content(full_prompt)
            if response and response.text:
                subject, body = self.parse_email_response(response.text)
                if subject and body:
                    # Preview email
                    print(f"\n📧 Generated Email:")
                    print(f"To: {recipient}")
                    print(f"Subject: {subject}")
                    print(f"Body: {body}")
                    print("-" * 50)
                    
                    # Ask for confirmation to send
                    send_choice = self.get_text_input("Would you like to send this email? (yes/no)")
                    
                    if send_choice.lower() in ['yes', 'y', 'send', 'sure', 'okay']:
                        success = self.send_email(recipient, subject, body)
                        if success:
                            self.speak("Great! Your email has been sent successfully!")
                        else:
                            self.speak("Sorry, there was an error sending the email. Please try again.")
                    else:
                        self.speak("Email cancelled. No worries!")
                else:
                    self.speak("Sorry, I couldn't generate a proper email. Let's try again.")
            else:
                self.speak("Sorry, I couldn't generate the email content. Let's try again.")
        except Exception as e:
            print(f"Error generating email: {e}")
            self.speak("Sorry, there was an error generating the email. Please try again.")
    
    def clean_email_address(self, email_text):
        """Clean up email address"""
        # Replace common spoken words with email symbols
        email_text = email_text.replace(' at ', '@')
        email_text = email_text.replace(' dot ', '.')
        email_text = email_text.replace(' underscore ', '_')
        email_text = email_text.replace(' dash ', '-')
        email_text = email_text.replace(' ', '')  # Remove spaces
        return email_text
    
    def generate_response(self, user_input):
        """Generate AI response to user input"""
        try:
            prompt = f"""
            You are a helpful Gmail AI assistant. The user said: "{user_input}"
            
            Provide a brief, helpful response. Keep it conversational and under 2 sentences.
            """
            
            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return None
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
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
                # Check if there's content on the same line
                body_content = line.replace("Body:", "").strip()
                if body_content:
                    body = body_content
            elif body_started:
                # Accumulate body content
                if body:
                    body += "\n" + line
                else:
                    body = line
        
        # Validation
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
    
    def send_email(self, to_email, subject, body, from_email="me"):
        """Send email directly"""
        print(f"Sending email to: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {body[:100]}...")
        
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
    import sys
    try:
        agent = VoiceGmailAIAgent(credentials_file='credentials.json')
        mode = input("Type 'voice' for voice mode, or press Enter for text mode: ").strip().lower()
        if mode == 'voice':
            agent.voice_interaction()
        else:
            agent.text_interaction()
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
