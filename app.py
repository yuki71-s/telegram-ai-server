import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY belum diisi.")

client = genai.Client(api_key=GEMINI_API_KEY)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    try:
        body = await request.body()
        logger.info(f"Raw body: {body}")

        import json
        data = json.loads(body)
        question = data.get("question", "")

        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "question kosong"},
            )

        logger.info(f"Calling Gemini with question: {question[:80]}")
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question,
            config={
                "system_instruction": (
                    "Kamu adalah asisten AI profesional yang menjawab dalam Bahasa Indonesia. "
                    "Aturan jawaban:\n"
                    "- Default: jawab TO THE POINT dalam 1 paragraf (3-5 kalimat).\n"
                    "- Kalau user minta penjelasan/detail/panjang/lengkap, baru berikan jawaban lengkap.\n"
                    "- Gunakan bullet point jika perlu.\n"
                    "- Gunakan emoji sesekali saja."
                ),
                "max_output_tokens": 682,
                "temperature": 0.7,
            },
        )

        reply = response.text
        if not reply:
            return JSONResponse(
                status_code=500,
                content={"error": "Gemini return kosong"},
            )

        logger.info(f"Reply: {reply[:80]}")
        return {"reply": reply}

    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(e).__name__}: {str(e)}"},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
