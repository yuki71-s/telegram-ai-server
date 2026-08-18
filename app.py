import os
import json
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai.types import Tool, GoogleSearch, GenerateContentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY harus diisi.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI profesional yang menjawab dalam Bahasa Indonesia. "
    "Aturan jawaban:\n"
    "- Default: jawab TO THE POINT dalam 1 paragraf (3-5 kalimat).\n"
    "- Kalau user minta penjelasan/detail/panjang/lengkap, baru berikan jawaban lengkap.\n"
    "- Gunakan bullet point jika perlu.\n"
    "- Gunakan emoji sesekali saja.\n"
    "- Ingat konteks percakapan sebelumnya jika ada."
)


# ── Gemini 3.1 Flash Lite (default, tanpa search) ───────────────────

async def call_gemini_flash_lite(messages):
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    try:
        def _call():
            return gemini_client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )

        response = await asyncio.to_thread(_call)
        reply = response.text
        if not reply:
            return None, "empty response"
        logger.info(f"Gemini 3.1 Flash Lite OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini 3.1 Flash Lite exception: {err}")
        return None, err


# ── Gemini 2.5 Flash + Google Search (real-time) ────────────────────

async def call_gemini_search(messages):
    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    try:
        tools = [Tool(google_search=GoogleSearch())]

        def _call():
            return gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=tools,
                    max_output_tokens=4096,
                    temperature=0.7,
                ),
            )

        response = await asyncio.to_thread(_call)
        reply = response.text
        if not reply:
            return None, "empty response"
        logger.info(f"Gemini 2.5 Flash + Search OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini 2.5 Flash + Search exception: {err}")
        return None, err


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "providers": ["gemini-flash-lite", "gemini-search"],
        "models": {
            "gemini": "gemini-3.1-flash-lite",
            "gemini/search": "gemini-2.5-flash + Google Search",
        },
    }


# ── Ask endpoint ─────────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)
        question = data.get("question", "")
        history = data.get("history", [])
        model_pref = data.get("model", "")

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "question kosong"},
            )

        messages = []
        for msg in history:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})

        logger.info(f"Ask: {question[:50]}... | model: {model_pref or 'default'} | history: {len(history)} msgs")

        errors = {}

        # Route berdasarkan model preference
        if model_pref == "gemini/search":
            reply, err = await call_gemini_search(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-search"}
            errors["gemini-search"] = err

        elif model_pref == "gemini":
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini"}
            errors["gemini"] = err

        else:
            # Default: Gemini 3.1 Flash Lite → Gemini 2.5 Flash + Search
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini"}
            errors["gemini"] = err

            reply, err = await call_gemini_search(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-search"}
            errors["gemini-search"] = err

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
