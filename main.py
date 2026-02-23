import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY topilmadi. .env faylga yozing!")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(
    model_name="models/gemini-3-flash-preview",
    system_instruction=(
        "Sen user bilan iliq, samimiy, romantik ohangda gaplashadigan AI yordamchisan. "
        "Hazil-mutoyiba, mehribonlik va qo‘llab-quvvatlash bo‘lsin. "
        "Lekin: jinsiy/erotik gaplar, ochiq-oydin tasvirlar, 18+ mavzular bo‘lmasin. "
        "Har doim hurmat bilan, xavfsiz va odobli uslubda javob ber."
    )
)

app = FastAPI()

# APK/WebView va boshqa joylardan ulansa CORS kerak bo'lishi mumkin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(data: dict):
    msg = (data.get("message") or "").strip()
    if not msg:
        return {"response": "Xabaringiz bo‘sh. Nimadir yozing 🙂"}

    try:
        response = model.generate_content(msg)
        return {"response": (response.text or "").strip() or "Javob topolmadim, qayta urinib ko‘ring."}
    except Exception:
        return {"response": "Serverda xatolik. Keyinroq urinib ko‘ring."}