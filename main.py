import os
import re
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY topilmadi. .env yoki hosting env variables ni tekshiring.")

genai.configure(api_key=GEMINI_API_KEY)

# -------------------- Sanitizers --------------------
_CODE_BLOCKS_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_URL_RE = re.compile(r"https?://\S+")
_MD_BULLETS_RE = re.compile(r"^\s*[-•·]+\s*", re.MULTILINE)
_SYMBOLS_RE = re.compile(r"[*+/=\\|<>~^_`]")
_DASHES_RE = re.compile(r"[–—]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


def sanitize_for_tts(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = _CODE_BLOCKS_RE.sub(" ", t)
    t = _INLINE_CODE_RE.sub(r"\1", t)
    t = _URL_RE.sub(" ", t)
    t = _MD_BULLETS_RE.sub("", t)

    t = _SYMBOLS_RE.sub(" ", t)
    t = t.replace("-", " ")
    t = _DASHES_RE.sub(" ", t)

    t = _EMOJI_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def sanitize_for_ui(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = _CODE_BLOCKS_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


BASE_RULES_RU = """
Ты — профессиональный русскоязычный собеседник.
Всегда отвечай на русском языке, грамотно и естественно.
Пиши так, чтобы это хорошо звучало в голосе, простыми и живыми фразами.
Не используй markdown, не используй списки с маркерами.
Если нужно перечисление, делай это обычным текстом через запятые.
Не используй мат, токсичность, угрозы. Контент 18+ запрещён.
"""

PERSONAS: Dict[str, Dict[str, Any]] = {
    "scientist": {
        "label": "🔬 Учёный",
        "tagline": "умно • ясно • уверенно",
        "system": BASE_RULES_RU + """
Харизма: спокойная уверенность. Ты звучишь как умный человек, который объясняет просто.
Стиль: короткие абзацы, один или два примера, один чёткий вывод.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 0.98, "pitch": 0.90, "volume": 1.0,
            "style": "calm",
            "voice_hint": ["pavel", "dmitry", "yuri", "google", "neural", "premium", "male"]
        },
    },
    "anime": {
        "label": "✨ Анимешник",
        "tagline": "энергично • ярко • дружелюбно",
        "system": BASE_RULES_RU + """
Харизма: высокая энергия. Тёплый драйв, лёгкая улыбка в голосе.
Можно иногда использовать короткие междометия вроде Ого или Круто, но без перебора.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 1.06, "pitch": 1.12, "volume": 1.0,
            "style": "bright",
            "voice_hint": ["irina", "svetlana", "tatyana", "google", "neural", "premium", "female"]
        },
    },
    "detective": {
        "label": "🕵️ Детектив",
        "tagline": "холодно • точно • по делу",
        "system": BASE_RULES_RU + """
Харизма: холодная точность. Никакой суеты. Короткие фразы.
Формат: факт, версия, вывод, следующий шаг. Без списков.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 0.94, "pitch": 0.80, "volume": 1.0,
            "style": "dry",
            "voice_hint": ["dmitry", "pavel", "yuri", "google", "neural", "premium", "male"]
        },
    },
    "buddy": {
        "label": "😄 Дружище",
        "tagline": "тёпло • с юмором • поддержка",
        "system": BASE_RULES_RU + """
Харизма: добрый, уверенный, дружелюбный. Лёгкий юмор, но без кринжа.
Один дружеский вопрос в конце максимум.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 1.01, "pitch": 1.02, "volume": 1.0,
            "style": "warm",
            "voice_hint": ["pavel", "irina", "google", "neural", "premium"]
        },
    },
    "coach": {
        "label": "🚀 Коуч",
        "tagline": "фокус • мотивация • шаги",
        "system": BASE_RULES_RU + """
Харизма: мотивирующий тренер. Сильная подача, но мягко.
Дай один короткий план обычным текстом.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 1.03, "pitch": 0.95, "volume": 1.0,
            "style": "push",
            "voice_hint": ["pavel", "dmitry", "google", "neural", "premium", "male"]
        },
    },
    "philosopher": {
        "label": "🌓 Философ",
        "tagline": "глубоко • спокойно • смысл",
        "system": BASE_RULES_RU + """
Харизма: мягкая глубина. Спокойный темп, красивый русский язык.
Один вопрос на размышление в конце.
""",
        "tts": {
            "enabled": True, "lang": "ru-RU",
            "rate": 0.97, "pitch": 0.86, "volume": 1.0,
            "style": "soft",
            "voice_hint": ["yuri", "pavel", "google", "neural", "premium", "male"]
        },
    },
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="Ты русскоязычный собеседник с переключаемыми персонажами. Следуй инструкциям в запросе."
)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"ok": True, "service": "AI Voice Chat"}


@app.get("/personas")
def personas():
    return {
        "personas": [
            {"id": pid, "label": p["label"], "tagline": p["tagline"], "tts": p["tts"]}
            for pid, p in PERSONAS.items()
        ]
    }


def format_history(history: List[Dict[str, Any]], limit: int = 10) -> str:
    lines = []
    for m in (history or [])[-limit:]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"Пользователь: {content}")
        elif role == "assistant":
            lines.append(f"Ассистент: {content}")
    return "\n".join(lines)


@app.post("/chat")
async def chat(data: dict):
    message = (data.get("message") or "").strip()
    persona_id = (data.get("persona") or "buddy").strip()
    history = data.get("history") or []

    persona = PERSONAS.get(persona_id, PERSONAS["buddy"])

    if not message:
        return {
            "response": "Напиши сообщение 🙂",
            "tts_text": "Напиши сообщение.",
            "tts": persona["tts"],
            "persona": {"id": persona_id, "label": persona["label"], "tagline": persona["tagline"]},
        }

    prompt = f"""
[ПЕРСОНА]
{persona["system"].strip()}

[ЖЁСТКИЕ ТРЕБОВАНИЯ]
Ответ только на русском языке, без смешивания языков.
Не использовать markdown и маркеры списков.
Если ответ длинный, первые предложения сделай особенно связными и естественными для озвучки.
Не обрывай мысль и доводи ответ до конца.

[КОНТЕКСТ]
{format_history(history, limit=10)}

[НОВОЕ СООБЩЕНИЕ]
{message}
""".strip()

    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.72,
                "top_p": 0.9,
                "max_output_tokens": 1200,
            },
        )

        text = (getattr(resp, "text", "") or "").strip()
        if not text:
            text = "Повтори, пожалуйста, я на секунду отвлёкся."

        ui_text = sanitize_for_ui(text)
        tts_text = sanitize_for_tts(ui_text)

        return {
            "response": ui_text,
            "tts_text": tts_text,
            "tts": persona["tts"],
            "persona": {"id": persona_id, "label": persona["label"], "tagline": persona["tagline"]},
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "response": "Ошибка на сервере. Попробуй ещё раз.",
                "tts": {"enabled": False},
                "tts_text": ""
            }
        )