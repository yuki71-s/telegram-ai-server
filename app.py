import os
import json
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ── API Keys ────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GROQ_API_KEY and not CEREBRAS_API_KEY and not GEMINI_API_KEY:
    raise ValueError("Minimal 1 API key harus diisi (GROQ/CEREBRAS/GEMINI).")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI profesional yang menjawab dalam Bahasa Indonesia. "
    "Aturan jawaban:\n"
    "- Default: jawab TO THE POINT dalam 1 paragraf (3-5 kalimat).\n"
    "- Kalau user minta penjelasan/detail/panjang/lengkap, baru berikan jawaban lengkap.\n"
    "- Gunakan bullet point jika perlu.\n"
    "- Gunakan emoji sesekali saja.\n"
    "- Ingat konteks percakapan sebelumnya jika ada."
)


# ── Provider: Groq ─────────────────────────────────────────────────
async def call_groq(messages: list[dict]) -> str | None:
    if not GROQ_API_KEY:
        return None

    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        groq_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": groq_messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning("Groq rate limit (429)")
            return None

        if resp.status_code != 200:
            logger.error(f"Groq error: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        logger.info(f"Groq reply: {reply[:80]}")
        return reply

    except Exception as e:
        logger.error(f"Groq error: {type(e).__name__}: {e}")
        return None


# ── Provider: Cerebras ─────────────────────────────────────────────
async def call_cerebras(messages: list[dict]) -> str | None:
    if not CEREBRAS_API_KEY:
        return None

    cerebras_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        cerebras_messages.append({"role": msg["role"], "content": msg["content"]})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b",
                    "messages": cerebras_messages,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning("Cerebras rate limit (429)")
            return None

        if resp.status_code != 200:
            logger.error(f"Cerebras error: {resp.status_code} {resp.text[:200]}")
            return None

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        logger.info(f"Cerebras reply: {reply[:80]}")
        return reply

    except Exception as e:
        logger.error(f"Cerebras error: {type(e).__name__}: {e}")
        return None


# ── Provider: Gemini 2.5 Flash ─────────────────────────────────────
async def call_gemini(messages: list[dict]) -> str | None:
    if not gemini_client:
        return None

    contents = []
    for msg in messages:
        role = msg["role"]
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    try:
        import asyncio

        def _call():
            return gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "max_output_tokens": 4096,
                    "temperature": 0.7,
                },
            )

        response = await asyncio.to_thread(_call)
        reply = response.text
        if not reply:
            return None
        logger.info(f"Gemini reply: {reply[:80]}")
        return reply

    except Exception as e:
        logger.error(f"Gemini error: {type(e).__name__}: {e}")
        return None


# ── Health ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    providers = []
    if GROQ_API_KEY:
        providers.append("groq")
    if CEREBRAS_API_KEY:
        providers.append("cerebras")
    if GEMINI_API_KEY:
        providers.append("gemini")
    return {"status": "ok", "providers": providers}


# ── Ask (with fallback) ────────────────────────────────────────────
@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)
        question = data.get("question", "")
        history = data.get("history", [])

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "question kosong"},
            )

        messages = []
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})

        logger.info(f"Ask: {question[:50]}... | history: {len(history)} msgs")

        # Fallback order: Groq → Cerebras → Gemini
        reply = await call_groq(messages)
        if reply:
            return {"reply": reply, "provider": "groq"}

        logger.info("Groq failed, trying Cerebras...")
        reply = await call_cerebras(messages)
        if reply:
            return {"reply": reply, "provider": "cerebras"}

        logger.info("Cerebras failed, trying Gemini...")
        reply = await call_gemini(messages)
        if reply:
            return {"reply": reply, "provider": "gemini"}

        return JSONResponse(
            status_code=503,
            content={"error": "Semua provider gagal. Coba lagi nanti."},
        )

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
