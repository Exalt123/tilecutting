# 🔷 Technema Flow Optimizer

> Minimize changeovers. Maximize production efficiency.

A web-based application that uses AI (Google Gemini 1.5 Flash) to intelligently optimize tile cutting production flow for the Technima cutter.

## 🎯 The Problem

**Current State:** Production follows Job ID order, causing excessive machine changeovers.
```
Job A (2×2) → Job A (4×4) → Job B (2×2) = 2 Changeovers
```

**Optimized State:** Orders grouped by cut size minimize changeovers.
```
Job A (2×2) → Job B (2×2) → Job A (4×4) = 1 Changeover
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd my-tile-app
pip install -r requirements.txt
```

### 2. Configure Secrets
Edit `.streamlit/secrets.toml` with your credentials:
- `app_password` - Shop login password
- `gemini_api_key` - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- `spreadsheet_id` - Your Google Sheet ID
- `gcp_service_account` - Service account credentials

### 3. Run the App
```bash
streamlit run app.py
```

## 📋 Google Sheet Structure

Your Master Order Sheet should have these columns:

| Column | Description |
|--------|-------------|
| Job ID | Unique job identifier |
| Tile Color | Color/material description |
| Cut Width | Width dimension |
| Cut Height | Height dimension |
| Quantity | Number of tiles |
| Status | Leave blank for pending, app writes "Scheduled [date]" |

## 🔄 User Flow

1. **Login** → Enter shop password
2. **Refresh Data** → Pull pending orders from Google Sheets
3. **Optimize** → AI groups orders by cut size
4. **Approve** → Mark orders as "Scheduled" in the sheet

## 🧠 How the AI Works

The Gemini optimizer follows these rules:

**Hard Constraints:**
- Group identical cut sizes together (2×2 with 2×2)
- Preserve all order data
- Include every order

**Soft Constraints:**
- Smooth transitions between sizes (2×2 → 2×3 better than 2×2 → 24×48)
- Change one dimension at a time when possible
- Generally order small → large

## 📁 Project Structure

```
my-tile-app/
├── .streamlit/
│   └── secrets.toml       # API keys & credentials (gitignored)
├── services/
│   ├── __init__.py
│   ├── gsheets_service.py # Google Sheets read/write
│   └── gemini_service.py  # AI optimization logic
├── app.py                 # Main Streamlit UI
├── requirements.txt       # Dependencies
└── README.md
```

## 🔐 Security

- Password-protected login
- Credentials stored in Streamlit secrets (not in code)
- `.gitignore` prevents accidental credential commits

## 📊 Technical Stack

| Component | Tool | Why |
|-----------|------|-----|
| Frontend | Streamlit | Fast Python web apps |
| Backend | Python 3.9+ | Data manipulation |
| AI Model | Gemini 1.5 Flash | Fast, free tier available |
| Database | Google Sheets | Zero-cost, team-friendly |
| Hosting | Streamlit Cloud | Free SSL & deployment |

## 🚀 Deployment to Streamlit Cloud

1. Push code to GitHub (secrets.toml will NOT be committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repo
4. Add secrets in App Settings → Secrets
5. Share the URL with your team!

---

© 2026 Exalt Samples LLC
