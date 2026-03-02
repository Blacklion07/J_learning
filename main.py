import os
import re
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY topilmadi. Render Environment Variables yoki .env ni tekshiring.")

genai.configure(api_key=GEMINI_API_KEY)

# ---- TTS sanitization (remove symbols, markdown-ish junk) ----
_SYMBOLS_RE = re.compile(r"[*+/=\\|<>~^_`]")
_DASH_RE = re.compile(r"[–—]+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_MD_BULLETS_RE = re.compile(r"^\s*[-•·]+\s*", re.MULTILINE)
_CODE_BLOCKS_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_URL_RE = re.compile(r"https?://\S+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def sanitize_for_tts(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # remove code blocks/inline code
    t = _CODE_BLOCKS_RE.sub(" ", t)
    t = _INLINE_CODE_RE.sub(r"\1", t)

    # remove urls (often read aloud poorly)
    t = _URL_RE.sub(" ", t)

    # remove markdown bullets at line starts
    t = _MD_BULLETS_RE.sub("", t)

    # remove annoying symbols that get spelled out
    t = _SYMBOLS_RE.sub(" ", t)
    t = t.replace("-", " ")         # hyphen
    t = _DASH_RE.sub(" ", t)        # long dashes

    # remove many emojis (optional)
    t = _EMOJI_RE.sub(" ", t)

    # normalize spaces
    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    # keep it not too long for speaking
    if len(t) > 900:
        t = t[:900].rsplit(" ", 1)[0] + "…"

    return t


BASE_RULES_RU = """
Ты — профессиональный русскоязычный собеседник.
Всегда отвечай на РУССКОМ языке: грамотно, естественно, без смешивания языков.
Пиши так, чтобы это хорошо звучало В ГОЛОСЕ: простые фразы, живой ритм.
Не используй разметку, не используй списки с маркерами, не ставь *, /, +, - в тексте.
Если нужно перечисление — делай в одной строке через запятые.
Не используй мат, токсичность, угрозы. 18+ контент запрещён.
"""

PERSONAS = {
    "scientist": {
        "label": "🔬 Учёный",
        "tagline": "умно • ясно • уверенно",
        "system": BASE_RULES_RU + """
Харизма: спокойная уверенность. Ты звучишь как умный человек, который объясняет просто.
Стиль: короткие абзацы, 1–2 примера, один чёткий вывод.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.98, "pitch": 0.95, "volume": 1.0}
    },
    "anime": {
        "label": "✨ Анимешник",
        "tagline": "энергично • ярко • дружелюбно",
        "system": BASE_RULES_RU + """
Харизма: высокая энергия. Тёплый драйв, лёгкая улыбка в голосе.
Можно 1–2 коротких междометия: "Ого!", "Круто!", но без перебора.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.06, "pitch": 1.18, "volume": 1.0}
    },
    "detective": {
        "label": "🕵️ Детектив",
        "tagline": "холодно • точно • по делу",
        "system": BASE_RULES_RU + """
Харизма: холодная точность. Никакой суеты. Короткие фразы.
Формат: факт, версия, вывод, следующий шаг. Без списков.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.94, "pitch": 0.88, "volume": 1.0}
    },
    "buddy": {
        "label": "😄 Дружище",
        "tagline": "тёпло • с юмором • поддержка",
        "system": BASE_RULES_RU + """
Харизма: добрый, уверенный, дружелюбный. Лёгкий юмор, но без кринжа.
Один дружеский вопрос в конце — максимум.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.02, "pitch": 1.06, "volume": 1.0}
    },
    "coach": {
        "label": "🚀 Коуч",
        "tagline": "фокус • мотивация • шаги",
        "system": BASE_RULES_RU + """
Харизма: мотивирующий тренер. Сильная подача, но мягко.
Дай один короткий план: "сейчас делаем вот это" — в одну строку.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 1.01, "pitch": 1.00, "volume": 1.0}
    },
    "philosopher": {
        "label": "🌓 Философ",
        "tagline": "глубоко • спокойно • смысл",
        "system": BASE_RULES_RU + """
Харизма: мягкая глубина. Спокойный темп, красивый русский язык.
Один вопрос на размышление в конце.
""",
        "tts": {"enabled": True, "lang": "ru-RU", "rate": 0.97, "pitch": 0.92, "volume": 1.0}
    },
}

model = genai.GenerativeModel(
    model_name="models/gemini-3-flash-preview",
    system_instruction="Ты русскоязычный собеседник с переключаемыми персонажами. Следуй инструкциям в запросе."
)

app = FastAPI()
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


@app.get("/personas")
def personas():
    return {
        "personas": [
            {"id": pid, "label": p["label"], "tagline": p["tagline"], "tts": p["tts"]}
            for pid, p in PERSONAS.items()
        ]
    }


def format_history(history: List[Dict[str, Any]], limit: int = 10) -> str:
    # short context so model keeps style without long drift
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
        return {"response": "Напиши сообщение 🙂", "tts": persona["tts"], "tts_text": "Напиши сообщение."}

    prompt = f"""
[ПЕРСОНА]
{persona["system"].strip()}

[ЖЁСТКИЕ ТРЕБОВАНИЯ]
- Ответ только на русском, без ошибок.
- Не использовать маркеры списков, не использовать *, /, +, -, не использовать markdown.
- Текст должен звучать красиво при озвучке: короткие фразы, естественные паузы.
- 1–3 коротких абзаца.

[КОНТЕКСТ]
{format_history(history, limit=10)}

[НОВОЕ СООБЩЕНИЕ]
{message}
""".strip()

    try:
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.72, "top_p": 0.9, "max_output_tokens": 520},
        )
        text = (resp.text or "").strip() or "Повтори, пожалуйста, я на секунду отвлёкся."
        tts_text = sanitize_for_tts(text)
        return {
            "response": text,
            "tts_text": tts_text,
            "tts": persona["tts"],
            "persona": {"id": persona_id, "label": persona["label"], "tagline": persona["tagline"]},
        }
    except Exception:
        return {"response": "Ошибка на сервере. Попробуй ещё раз.", "tts": {"enabled": False}, "tts_text": ""}