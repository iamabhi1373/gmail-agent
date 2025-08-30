#!/usr/bin/env python3
"""
Enhanced Gmail Automation AI Agent
- Processes HR contacts PDF
- Generates personalized emails using Gemini
- Sends emails via Gmail API with rate limiting
"""

import os
import time
import base64
import logging
import mimetypes
from email.message import EmailMessage
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import argparse
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Google API imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Gemini import
import google.generativeai as genai

# Scopes for Gmail API
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

@dataclass
class SendResult:
    total: int
    success: int
    failed: int

class EnhancedGmailAIAgent:
    def __init__(self, credentials_file: str = 'credentials.json') -> None:
        load_dotenv()
        self.credentials_file = credentials_file
        self.service = self._authenticate_gmail()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.resume_path = os.getenv("RESUME_PATH", "resume.pdf")
        self.email_delay_seconds = int(os.getenv("EMAIL_DELAY_SECONDS", "30"))
        self.max_emails_per_hour = int(os.getenv("MAX_EMAILS_PER_HOUR", "50"))
        # Email style controls
        self.email_tone = os.getenv("EMAIL_TONE", "professional")  # professional | friendly | warm
        self.email_words = int(os.getenv("EMAIL_WORDS", "150"))     # approximate word target
        self.email_subject_prefix = os.getenv("EMAIL_SUBJECT_PREFIX", "")
        self.email_signature = os.getenv("EMAIL_SIGNATURE", "Best regards,\nAbhishek Kumar Gupta")
        self.email_use_template = os.getenv("EMAIL_USE_TEMPLATE", "false").lower() in {"1","true","yes"}
        self.email_fallback_on_ai_error = os.getenv("EMAIL_FALLBACK_ON_AI_ERROR", "true").lower() in {"1","true","yes"}
        # Default generalized SDE interest template (short HR-friendly)
        self._sde_template_subject = "Software Engineer opportunities"
        self._sde_template_body = (
            "Dear Hiring Team,\n\n"
            "I’m interested in Software Engineer roles at your organization. I’ve built reliable, scalable "
            "services, shipped features end‑to‑end, and improved developer velocity through automation and "
            "strong testing/observability.\n\n"
            "I’ve attached my resume. Could we schedule a brief 15‑minute chat to discuss potential fit?\n\n"
            "{signature}"
        )

    def _authenticate_gmail(self):
        creds = None
        token_path = 'token.json'
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        return build('gmail', 'v1', credentials=creds)

    def process_hr_pdf(self, pdf_path: str) -> pd.DataFrame:
        import pdfplumber
        rows = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                headers = [h.strip() if h else '' for h in table[0]]
                for r in table[1:]:
                    row = {headers[i] if i < len(headers) else f"col_{i}": (r[i] or '').strip() for i in range(len(r))}
                    rows.append(row)
        df = pd.DataFrame(rows)
        # Normalize expected columns
        rename_map = {}
        for col in df.columns:
            lc = col.lower()
            if 'name' in lc and 'company' not in lc:
                rename_map[col] = 'Name'
            elif 'email' in lc:
                rename_map[col] = 'Email'
            elif 'title' in lc or 'position' in lc:
                rename_map[col] = 'Title'
            elif 'company' in lc:
                rename_map[col] = 'Company'
        df = df.rename(columns=rename_map)
        # Keep only relevant fields
        keep = [c for c in ['Name','Email','Title','Company'] if c in df.columns]
        df = df[keep]
        # Drop rows without email
        df = df.dropna(subset=['Email']) if 'Email' in df.columns else pd.DataFrame(columns=['Name','Email','Title','Company'])
        return df

    def _build_model_prompt(self, hr_row: Dict[str, str]) -> str:
        name = hr_row.get('Name', 'there')
        title = hr_row.get('Title', '')
        company = hr_row.get('Company', '')
        tone = self.email_tone
        words = self.email_words
        signature = self.email_signature
        return (
            "You are an expert recruiter outreach writer. "
            f"Write a concise, {tone} cold email to {name} ({title}) at {company}. "
            "Purpose: explore opportunities and request a brief chat. "
            "Mention that my resume is attached (do not add links). "
            f"Target length: ~{words} words. Avoid fluff and generic claims. "
            "Return EXACTLY two sections in plain text:\n"
            "Subject: <short compelling subject>\n"
            "Body: <email body with greeting, 2-3 tight paragraphs, a clear CTA, and this signature appended>\n"
            f"Signature to append verbatim after a blank line:\n{signature}"
        )

    def _parse_model_output(self, text: str) -> Tuple[str, str]:
        subject = "Opportunity to connect"
        body = text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        subj_line_idx = next((i for i,l in enumerate(lines) if l.lower().startswith("subject:")), -1)
        body_line_idx = next((i for i,l in enumerate(lines) if l.lower().startswith("body:")), -1)
        if subj_line_idx != -1:
            subj_line = lines[subj_line_idx]
            subject_val = subj_line.split(":",1)[1].strip() if ":" in subj_line else ""
            if subject_val:
                subject = subject_val
        if body_line_idx != -1:
            body_lines = []
            for l in lines[body_line_idx+1:]:
                if l.lower().startswith("subject:"):
                    break
                body_lines.append(l)
            # Some models keep the body text on the same line after Body:
            if not body_lines and body_line_idx != -1:
                inline = lines[body_line_idx]
                after = inline.split(":",1)[1].strip() if ":" in inline else ""
                if after:
                    body_lines = [after]
            if body_lines:
                body = "\n".join(body_lines).strip()
        # Apply optional subject prefix
        if self.email_subject_prefix:
            pref = self.email_subject_prefix.strip()
            if pref and not subject.lower().startswith(pref.lower()):
                subject = f"{pref} {subject}".strip()
        return subject, body

    def _generate_template_email(self, hr_row: Dict[str, str]) -> Tuple[str, str]:
        name = hr_row.get('Name') or 'there'
        company = hr_row.get('Company') or 'your company'
        subject = self._sde_template_subject.format(company=company)
        body = self._sde_template_body.format(
            name=name,
            company=company,
            signature=self.email_signature
        )
        if self.email_subject_prefix:
            pref = self.email_subject_prefix.strip()
            if pref and not subject.lower().startswith(pref.lower()):
                subject = f"{pref} {subject}".strip()
        return subject, body

    def generate_personalized_email(self, hr_row: Dict[str, str]) -> Tuple[str, str]:
        if self.email_use_template:
            return self._generate_template_email(hr_row)
        # AI generation with safe fallback
        try:
            prompt = self._build_model_prompt(hr_row)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
            return self._parse_model_output(text)
        except Exception as e:
            logging.warning("AI generation failed (%s). Falling back to template.", e)
            if self.email_fallback_on_ai_error:
                return self._generate_template_email(hr_row)
            raise

    def _create_message(self, to_email: str, subject: str, body: str, attachment_path: Optional[str] = None) -> Dict:
        message = EmailMessage()
        message["To"] = to_email
        message["Subject"] = subject
        message["From"] = "me"
        message.set_content(body)
        if attachment_path and os.path.exists(attachment_path):
            mime_type, _ = mimetypes.guess_type(attachment_path)
            maintype, subtype = (mime_type or 'application/pdf').split('/')
            with open(attachment_path, 'rb') as f:
                message.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {"raw": encoded_message}

    def send_single_email(self, hr_row: Dict[str, str]) -> bool:
        to_email = hr_row.get('Email')
        if not to_email:
            return False
        subject, body = self.generate_personalized_email(hr_row)
        msg = self._create_message(to_email, subject, body, self.resume_path)
        try:
            self.service.users().messages().send(userId='me', body=msg).execute()
            logging.info("Sent email to %s", to_email)
            return True
        except Exception as e:
            logging.error("Failed to send to %s: %s", to_email, e)
            return False

    def send_bulk_emails(self, pdf_path: str, max_emails: Optional[int] = None) -> Dict[str, int]:
        df = self.process_hr_pdf(pdf_path)
        total = 0
        success = 0
        failed = 0
        for _, row in df.iterrows():
            if max_emails is not None and total >= max_emails:
                break
            total += 1
            ok = self.send_single_email(row.to_dict())
            if ok:
                success += 1
            else:
                failed += 1
            time.sleep(self.email_delay_seconds)
        return {"total": total, "success": success, "failed": failed}

    def fetch_hr_names_and_companies(self, pdf_path: str) -> pd.DataFrame:
        """Return a DataFrame with at least Name and Company columns parsed from the PDF.
        Falls back to empty strings if a column is missing.
        """
        df = self.process_hr_pdf(pdf_path)
        # Ensure columns exist
        if 'Name' not in df.columns:
            df['Name'] = ''
        if 'Company' not in df.columns:
            df['Company'] = ''
        return df[[c for c in ['Name', 'Company', 'Email', 'Title'] if c in df.columns]].copy()

def _setup_logging() -> None:
    os.makedirs('logs', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logfile = os.path.join('logs', f'run_{ts}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

def _preview_emails(agent: EnhancedGmailAIAgent, df: pd.DataFrame, limit: int = 3) -> None:
    count = 0
    for _, row in df.iterrows():
        hr = row.to_dict()
        subject, body = agent.generate_personalized_email(hr)
        print("="*60)
        print(f"To: {hr.get('Email','(missing)')} | Name: {hr.get('Name','')} | Company: {hr.get('Company','')}")
        print(f"Subject: {subject}")
        print("-"*60)
        print(body[:800])
        count += 1
        if count >= limit:
            break

def main_cli():
    parser = argparse.ArgumentParser(description='Enhanced Gmail Automation')
    parser.add_argument('--pdf', default='hr_contacts.pdf', help='Path to HR contacts PDF')
    parser.add_argument('--max-emails', type=int, default=None, help='Max emails to send')
    parser.add_argument('--dry-run', action='store_true', help='Preview emails without sending')
    parser.add_argument('--preview', type=int, default=3, help='Number of previews to show in dry-run')
    parser.add_argument('--use-template', action='store_true', help='Force using built-in template')
    parser.add_argument('--no-template', action='store_true', help='Disable template and use AI generation')
    args = parser.parse_args()

    _setup_logging()

    agent = EnhancedGmailAIAgent()
    if args.use_template:
        agent.email_use_template = True
    if args.no_template:
        agent.email_use_template = False

    df = agent.process_hr_pdf(args.pdf)
    logging.info('Parsed %d rows from %s', len(df), args.pdf)

    if args.dry_run:
        _preview_emails(agent, df, args.preview)
        print("\nDry-run complete. No emails were sent.")
        return

    results = agent.send_bulk_emails(args.pdf, max_emails=args.max_emails)
    print("\n" + "="*40)
    print("BULK SEND SUMMARY")
    print("="*40)
    print(f"Total processed: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")

if __name__ == "__main__":
    main_cli()


    
