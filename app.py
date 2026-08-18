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

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GROQ_API_KEY and not CEREBRAS_API_KEY and not GEMINI_API_KEY:
    raise ValueError("Minimal 1 API key harus diisi.")

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


def build_openai_messages(messages):
    result = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        role = msg.get("role", "user")
        if role == "model":
            role = "assistant"
        result.append({"role": role, "content": msg.get("content", "")})
    return result


async def call_groq(messages):
    if not GROQ_API_KEY:
        return None, "no key"

    oai_msgs = build_openai_messages(messages)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": oai_msgs,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning(f"Groq rate limit: {resp.text[:200]}")
            return None, "429 rate limit"

        if resp.status_code != 200:
            err = resp.text[:200]
            logger.error(f"Groq {resp.status_code}: {err}")
            return None, f"{resp.status_code}: {err}"

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        logger.info(f"Groq OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Groq exception: {err}")
        return None, err


async def call_cerebras(messages):
    if not CEREBRAS_API_KEY:
        return None, "no key"

    oai_msgs = build_openai_messages(messages)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-oss-120b",
                    "messages": oai_msgs,
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning(f"Cerebras rate limit: {resp.text[:200]}")
            return None, "429 rate limit"

        if resp.status_code != 200:
            err = resp.text[:200]
            logger.error(f"Cerebras {resp.status_code}: {err}")
            return None, f"{resp.status_code}: {err}"

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        logger.info(f"Cerebras OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Cerebras exception: {err}")
        return None, err


async def call_gemini(messages):
    if not gemini_client:
        return None, "no client"

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

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
            return None, "empty response"
        logger.info(f"Gemini OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini exception: {err}")
        return None, err


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

        errors = {}

        reply, err = await call_groq(messages)
        if reply:
            return {"reply": reply, "provider": "groq"}
        errors["groq"] = err

        reply, err = await call_cerebras(messages)
        if reply:
            return {"reply": reply, "provider": "cerebras"}
        errors["cerebras"] = err

        reply, err = await call_gemini(messages)
        if reply:
            return {"reply": reply, "provider": "gemini"}
        errors["gemini"] = err

        logger.error(f"All providers failed: {errors}")
        return JSONResponse(
            status_code=503,
            content={"error": "all providers failed", "details": errors},
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
