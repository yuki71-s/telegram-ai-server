# AI Server

Server AI untuk Telegram Bot menggunakan Google Gemini API.

## Deploy ke Render

1. Push repo ini ke GitHub
2. Login ke Render (https://dashboard.render.com)
3. Klik **+ New** → **Web Service**
4. Pilih repo GitHub ini
5. Isi setting:
   - **Name:** `telegram-ai-server`
   - **Language:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free`
6. Tambah Environment Variable:
   - `GEMINI_API_KEY` = (API key dari Google AI Studio)
7. Klik **Create Web Service**

## Endpoint

- `GET /health` - Health check
- `POST /ask` - Kirim pertanyaan ke AI

## Keep-Alive Setup

Setelah deploy, setup cron di https://cron-job.org:
- URL: `https://your-app-name.onrender.com/health`
- Schedule: Every 10 minutes
- Method: GET
