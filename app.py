import os
import json
import logging
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai.types import GenerateContentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

if not GEMINI_API_KEY and not OPENROUTER_API_KEY:
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


# ── Gemini 3.1 Flash Lite (default, cepat) ──────────────────────────

async def call_gemini_flash_lite(messages):
    if not gemini_client:
        return None, "no client"

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


# ── Gemini 3.6 Flash (pintar, fallback) ─────────────────────────────

async def call_gemini_flash(messages):
    if not gemini_client:
        return None, "no client"

    contents = []
    for msg in messages:
        role = msg.get("role", "user")
        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

    try:
        def _call():
            return gemini_client.models.generate_content(
                model="gemini-3.6-flash",
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
        logger.info(f"Gemini 3.6 Flash OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"Gemini 3.6 Flash exception: {err}")
        return None, err


# ── OpenRouter (text, image, web search) ────────────────────────────

async def call_openrouter(messages, model, image_url=None, video_url=None, web_search=False):
    if not OPENROUTER_API_KEY:
        return None, "no key"

    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        role = msg.get("role", "user")
        if role == "model":
            role = "assistant"
        elif role not in ("user", "assistant", "system"):
            role = "user"

        content_parts = []

        if image_url and role == "user" and msg == messages[-1]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        if video_url and role == "user" and msg == messages[-1]:
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": video_url},
            })

        content_parts.append({"type": "text", "text": msg.get("content", "")})
        oai_messages.append({"role": role, "content": content_parts})

    tools = []
    if web_search:
        tools.append({"type": "openrouter:web_search"})

    payload = {
        "model": model,
        "messages": oai_messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )

        if resp.status_code == 429:
            logger.warning(f"OpenRouter rate limit: {resp.text[:200]}")
            return None, "429 rate limit"

        if resp.status_code != 200:
            err = resp.text[:200]
            logger.error(f"OpenRouter {resp.status_code}: {err}")
            return None, f"{resp.status_code}: {err}"

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        if not reply:
            return None, "empty response"
        logger.info(f"OpenRouter ({model}) OK: {reply[:80]}")
        return reply, None

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"OpenRouter exception: {err}")
        return None, err


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    providers = []
    if GEMINI_API_KEY:
        providers.extend(["gemini-flash-lite", "gemini-flash"])
    if OPENROUTER_API_KEY:
        providers.append("openrouter")
    return {"status": "ok", "providers": providers}


# ── Ask endpoint ─────────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.body()
        data = json.loads(body)
        question = data.get("question", "")
        history = data.get("history", [])
        model_pref = data.get("model", "")
        image_url = data.get("image_url", "")
        video_url = data.get("video_url", "")
        web_search = data.get("web_search", False)

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "question kosong"},
            )

        messages = []
        for msg in history:
            role = msg.get("role", "user")
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})

        logger.info(f"Ask: {question[:50]}... | model: {model_pref or 'default'} | image: {bool(image_url)} | video: {bool(video_url)} | search: {web_search}")

        errors = {}

        # ── Video → OpenRouter vision model ──
        if video_url:
            vision_model = "google/gemma-4-26b-a4b-it:free"
            reply, err = await call_openrouter(messages, vision_model, video_url=video_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{vision_model}"}
            errors["openrouter-video"] = err

        # ── Gambar → langsung ke OpenRouter (vision model) ──
        elif image_url:
            vision_model = "google/gemma-4-26b-a4b-it:free"
            reply, err = await call_openrouter(messages, vision_model, image_url=image_url)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{vision_model}"}
            errors["openrouter-vision"] = err

        # ── Model preference routing ──
        elif model_pref.startswith("openrouter/"):
            or_model = model_pref.replace("openrouter/", "")
            reply, err = await call_openrouter(messages, or_model, web_search=web_search)
            if reply:
                return {"reply": reply, "provider": f"openrouter:{or_model}"}
            errors["openrouter"] = err

        elif model_pref == "gemini/flash":
            reply, err = await call_gemini_flash(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.6-flash"}
            errors["gemini-flash"] = err

        elif model_pref == "gemini":
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.1-flash-lite"}
            errors["gemini-flash-lite"] = err

        else:
            # Default: Gemini 3.1 Flash Lite → Gemini 3.6 Flash → OpenRouter
            reply, err = await call_gemini_flash_lite(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.1-flash-lite"}
            errors["gemini-flash-lite"] = err

            reply, err = await call_gemini_flash(messages)
            if reply:
                return {"reply": reply, "provider": "gemini-3.6-flash"}
            errors["gemini-flash"] = err

            if OPENROUTER_API_KEY:
                reply, err = await call_openrouter(messages, "deepseek/deepseek-v4-flash-0731", web_search=web_search)
                if reply:
                    return {"reply": reply, "provider": "openrouter:deepseek-v4-flash"}
                errors["openrouter"] = err

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
