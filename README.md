# 📧 Gmail Automation Agent with Gemini API

This project is a **Python-based automation tool** that integrates the **Gemini API** with **Gmail API** to automatically generate and send personalized emails.  

It reduces manual effort by handling bulk emails, smart content generation, and customization using **AI-powered text generation**.  

---

## 🚀 Features
- 🤖 **AI Email Generation**: Uses Gemini API to draft personalized emails.  
- 📬 **Gmail Integration**: Sends emails directly via Gmail API.  
- 🔒 **Secure Authentication**: OAuth2.0 flow for Gmail with token refresh.  
- ⚡ **Bulk Email Support**: Automate sending multiple emails in one run.  
- 🔑 **Environment Variable Support**: Store API keys securely using `.env`.  

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- [Google Gmail API](https://developers.google.com/gmail/api)  
- [Google Generative AI (Gemini API)](https://ai.google.dev/)  
- `google-auth`, `google-auth-oauthlib`, `google-api-python-client`  
- `python-dotenv` for environment variables  

---

## 📂 Project Structure
.
├── gmail_agent.py # Main automation script (Gmail + Gemini)
├── requirements.txt # Python dependencies
├── .env # Store API keys (not committed)
├── token.json # Generated automatically after first login (ignored)
├── credentials.json # OAuth2 credentials from Google Cloud (ignored)
└── README.md # Documentation
2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. Setup Google Cloud Project

Go to Google Cloud Console
.

Enable Gmail API.

Create OAuth client ID credentials.

Download credentials.json and place it in the project root.

⚠️ Do not commit credentials.json or token.json to GitHub. Add them to .gitignore.

5. Setup Environment Variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key_here

▶️ Usage

Run the script:

python gmail_agent.py
