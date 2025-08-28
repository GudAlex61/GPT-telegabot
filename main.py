#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64
import logging
import aiohttp
import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import Command, StateFilter
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeDefault, InlineKeyboardButton
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import LabeledPrice
from aiogram.filters import Command
import asyncio
from aiogram.utils.keyboard import InlineKeyboardBuilder
from PIL import Image
from typing import Optional, Dict, Any
from uuid import uuid4
import boto3
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor
from aiogram.types import BufferedInputFile
from html import escape as html_escape
import sqlite3
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from googletrans import Translator

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Load .env ----------
env_path = Path(".") / ".env"
if not env_path.exists():
    print("❌ .env not found. Create .env with TELEGRAM_BOT_TOKEN, OPENROUTER_API_KEY and S3 settings")
    exit(1)
load_dotenv(env_path)
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# S3 / R2 config
S3_ENABLED = os.getenv("S3_ENABLED", "1") == "1"
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "").strip()  # e.g. https://<accountid>.r2.cloudflarestorage.com
S3_REGION = os.getenv("S3_REGION", "") or None
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PRESIGNED_EXPIRATION = int(os.getenv("S3_PRESIGNED_EXPIRATION", "600"))  # seconds
S3_AUTO_DELETE_AFTER = int(os.getenv("S3_AUTO_DELETE_AFTER", "0"))  # seconds, 0 = disabled

# Compression settings
IMAGE_MAX_SIZE = int(os.getenv("IMAGE_MAX_SIZE", "1600"))  # px (largest side)
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "90"))  # JPEG quality (1-100)
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
TOGETHER_AI_API_URL = "https://api.together.xyz/v1/images/generations"

# Models that accept image_url (IDs)
MODELS_WITH_IMAGE_URL = {
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-exp:free"
}

PAYMENT_PROVIDER_TOKEN = os.getenv("PAYMENT_PROVIDER_TOKEN")

# Token costs
TOKEN_COST_TEXT = 1
TOKEN_COST_OCR = 3
TOKEN_COST_IMAGE_GEN = 5
STARTING_TOKENS = 30
REFERRAL_TOKENS = 30

# Basic checks
if not API_TOKEN or not OPENROUTER_API_KEY:
    logging.error("TELEGRAM_BOT_TOKEN or OPENROUTER_API_KEY missing in .env")
    exit(1)
if S3_ENABLED:
    if not (S3_ENDPOINT_URL and S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        logging.error("S3_ENABLED=1 but S3_ENDPOINT_URL/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET not fully set in .env")
        exit(1)

# ---------- Bot ----------
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
executor = ThreadPoolExecutor(max_workers=4)


# ---------- FSM States ----------
class ImageGenState(StatesGroup):
    waiting_for_prompt = State()


# ---------- Models ----------
AVAILABLE_MODELS = {
    "Deepseek-V3": {"id": "deepseek/deepseek-chat-v3-0324:free", "image_support": False},
    "Deepseek-R1": {"id": "deepseek/deepseek-r1-0528:free", "image_support": False},
    "Qwen3-coder": {"id": "qwen/qwen3-coder", "image_support": False},
    "Gemini-2.0-flash": {"id": "google/gemini-2.0-flash-001", "image_support": False},
    "Qwen3": {"id": "qwen/qwen3-14b:free", "image_support": False},
    "GPT-5 mini": {"id": "openai/gpt-4o-mini", "image_support": False},
}
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
OCR_MODEL = "google/gemini-flash-1.5"
user_models: Dict[int, str] = {}
last_bot_responses: Dict[int, str] = {}
last_user_message: Dict[int, str] = {}

# ---------- Database ----------
DB_NAME = "bot_users.db"


def init_db():
    """Создает таблицу пользователей, если её нет."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            referrer_id INTEGER,
            referral_count INTEGER DEFAULT 0,
            tokens INTEGER DEFAULT 30,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (referrer_id) REFERENCES users (user_id)
        )
    ''')
    # Создаем индекс для ускорения поиска по referrer_id
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_referrer_id ON users(referrer_id)')
    conn.commit()
    conn.close()
    logging.info("✅ База данных инициализирована")


def db_get_user(user_id: int) -> Optional[dict]:
    """Получает информацию о пользователе из БД."""
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row  # Позволяет обращаться к колонкам по имени
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logging.error(f"Ошибка при получении пользователя {user_id} из БД: {e}")
        return None


def db_create_user(user_id: int, referrer_id: Optional[int] = None) -> bool:
    """Создает нового пользователя в БД."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO users (user_id, referrer_id, tokens) VALUES (?, ?, ?)',
                       (user_id, referrer_id, STARTING_TOKENS))
        # Если пользователь успешно создан и у него есть реферер
        if cursor.rowcount > 0 and referrer_id:
            # Увеличиваем счетчик рефералов у пригласившего
            cursor.execute(
                'UPDATE users SET referral_count = referral_count + 1, tokens = tokens + ? WHERE user_id = ?',
                (REFERRAL_TOKENS, referrer_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Ошибка при создании пользователя {user_id} в БД: {e}")
        return False


def db_get_user_stats(user_id: int) -> dict:
    """Получает статистику пользователя."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT referral_count, tokens FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"referral_count": row[0], "tokens": row[1]}
        else:
            return {"referral_count": 0, "tokens": 0}
    except Exception as e:
        logging.error(f"Ошибка при получении статистики для {user_id}: {e}")
        return {"referral_count": 0, "tokens": 0}


def db_use_tokens(user_id: int, tokens: int) -> bool:
    """Списывает токены у пользователя. Возвращает True, если успешно."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Проверяем, достаточно ли токенов перед списанием
        cursor.execute('SELECT tokens FROM users WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        if row is None or row[0] < tokens:
            conn.close()
            return False  # Недостаточно токенов или пользователь не найден

        # Списываем токены
        cursor.execute('UPDATE users SET tokens = tokens - ? WHERE user_id = ?',
                       (tokens, user_id))
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    except Exception as e:
        logging.error(f"Ошибка при списании токенов у {user_id}: {e}")
        return False


# ---------- Utilities ----------
def convert_markdown_to_html(text: str) -> str:
    """
    Конвертирует markdown от нейросетей в HTML с правильным экранированием
    """
    if not text:
        return ""
    import re
    math_blocks_double = []  # Для $$...$$
    math_blocks_single = []  # Для $...$
    math_inlines = []  # Для $$...$$
    code_blocks = []

    # --- Обработка блочных формул ---
    def save_math_block_double(match):
        placeholder = f"{{MATH_BLOCK_DOUBLE_{len(math_blocks_double)}}}"
        math_content = match.group(1)
        math_content = replace_math_symbols(math_content)
        math_blocks_double.append(math_content)
        return placeholder

    # Обрабатываем $$...$$ (блочные формулы)
    text = re.sub(r'\$\$(.*?)\$\$', save_math_block_double, text, flags=re.DOTALL)
    # Обрабатываем $$...$$ (альтернативный формат для блоков)
    text = re.sub(r'\\\[(.*?)\\\]', save_math_block_double, text, flags=re.DOTALL)

    # --- Обработка inline формул с одиночным $ ---
    def save_math_block_single(match):
        placeholder = f"{{MATH_BLOCK_SINGLE_{len(math_blocks_single)}}}"
        math_content = match.group(1)
        math_content = replace_math_symbols(math_content)
        math_blocks_single.append(math_content)
        return placeholder

    text = re.sub(r'(?<!\\)(?<!\$)\$(?!\$)(.*?)(?<!\\)(?<!\$)\$(?!\$)', save_math_block_single, text, flags=re.DOTALL)

    # --- Обработка inline формул с двойным \( \) ---
    def save_math_inline(match):
        placeholder = f"{{MATH_INLINE_{len(math_inlines)}}}"
        math_content = match.group(1)
        math_content = replace_math_symbols(math_content)
        math_inlines.append(math_content)
        return placeholder

    text = re.sub(r'\\\((.*?)\\\)', save_math_inline, text, flags=re.DOTALL)

    # --- Обработка блоков кода ---
    def save_code_block(match):
        placeholder = f"{{CODE_BLOCK_{len(code_blocks)}}}"
        language = match.group(1) or ""
        code_content = match.group(2)
        code_blocks.append((language, code_content))
        return placeholder

    text = re.sub(r'```(\w+)\n(.*?)```', save_code_block, text, flags=re.DOTALL)
    text = re.sub(r'```\n(.*?)```', save_code_block, text, flags=re.DOTALL)
    text = text.replace('&', '&amp;').replace('<', '<').replace('>', '>')
    # --- Восстановление блоков ---
    for i, math_block in enumerate(math_blocks_double):
        text = text.replace(f'{{MATH_BLOCK_DOUBLE_{i}}}', f'<pre><code>{math_block}</code></pre>')
    for i, math_block in enumerate(math_blocks_single):
        text = text.replace(f'{{MATH_BLOCK_SINGLE_{i}}}', f'<code>{math_block}</code>')
    for i, math_inline in enumerate(math_inlines):
        text = text.replace(f'{{MATH_INLINE_{i}}}', f'<code>{math_inline}</code>')
    for i, (language, code_content) in enumerate(code_blocks):
        if language:
            text = text.replace(f'{{CODE_BLOCK_{i}}}',
                                f'<pre><code class="language-{language}">{code_content}</code></pre>')
        else:
            text = text.replace(f'{{CODE_BLOCK_{i}}}', f'<pre><code>{code_content}</code></pre>')
    # --- Обработка заголовков ---
    text = re.sub(r'^######\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#####\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^####\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    # --- Базовое форматирование ---
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\\)\*(?!\*)(.*?)(?<!\\)\*(?!\*)', r'<i>\1</i>', text)  # *...*
    text = re.sub(r'(?<!\\)_(?!_)(.*?)(?<!\\)_(?!_)', r'<i>\1</i>', text)  # _..._
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
    return text


def replace_math_symbols(text: str) -> str:
    import re
    text = re.sub(r'√\(-frac\{(.*?)\)\{(.*?)\}\}', lambda m: f"√(-({m.group(1)}/{m.group(2)}))", text)
    text = text.replace(r'\int', '∫')
    text = text.replace(r'\ln', 'ln')

    def sqrt_replace(match):
        content = match.group(1)
        return f"√({content})"

    text = re.sub(r'\\sqrt\{(.*?)\}', sqrt_replace, text)

    def frac_replace(match):
        numerator = match.group(1)
        denominator = match.group(2)
        return f"({numerator}/{denominator})"

    text = re.sub(r'\\frac\{(.*?)\}\{(.*?)\}', frac_replace, text)
    text = re.sub(r'-frac\{(.*?)\)\{(.*?)\}', lambda m: f"-({m.group(1)}/{m.group(2)})", text)
    text = re.sub(r'-frac\{(.*?)\}\{(.*?)\}', lambda m: f"-({m.group(1)}/{m.group(2)})", text)
    text = re.sub(r'\^2', '²', text)

    def text_replace(match):
        content = match.group(1)
        return content

    text = re.sub(r'\\text\{(.*?)\}', text_replace, text)
    text = text.replace(r'\quad', '  ')
    text = text.replace(r'\,', ' ')
    text = text.replace(r'\!', '')
    text = text.replace(r'\;', '')
    text = text.replace(r'\:', '')
    text = re.sub(r'\\left\s*\(', '(', text)
    text = re.sub(r'\\right\)', ')', text)
    text = re.sub(r'\\left\s*\[', '[', text)
    text = re.sub(r'\\right\]', ']', text)
    text = re.sub(r'\\left\s*\{', '{', text)
    text = re.sub(r'\\right\}', '}', text)
    text = re.sub(r'\\left\s*\|', '|', text)
    text = re.sub(r'\\right\s*\|', '|', text)
    text = re.sub(r'\\left\s*\\', '', text)
    text = re.sub(r'\\right\s*\\', '', text)
    text = re.sub(r'\\left\s*\.', '', text)
    text = re.sub(r'\\right\s*\.', '', text)
    text = re.sub(r'\\left\s*', '', text)
    text = re.sub(r'\\right\s*', '', text)

    def exp_replace(match):
        content = match.group(1)
        return f"e^({content})"

    text = re.sub(r'e\^\{(.*?)\}', exp_replace, text)

    def power_replace(match):
        content = match.group(1)
        return f"^({content})"

    text = re.sub(r'\^\{(.*?)\}', power_replace, text)

    def x_power_replace(match):
        content = match.group(1)
        return f"x^({content})"

    text = re.sub(r'x\^\{(.*?)\}', x_power_replace, text)
    text = text.replace(r'\infty', '∞')
    text = text.replace(r'\pm', '±')
    text = text.replace(r'\times', '×')
    text = text.replace(r'\div', '÷')
    text = text.replace(r'\leq', '≤')
    text = text.replace(r'\geq', '≥')
    text = text.replace(r'\neq', '≠')
    text = text.replace(r'\approx', '≈')
    text = text.replace(r'\cap', '∩')
    text = text.replace(r'\cup', '∪')
    text = text.replace(r'\subset', '⊂')
    text = text.replace(r'\subseteq', '⊆')
    text = text.replace(r'\in', '∈')
    text = text.replace(r'\notin', '∉')
    text = text.replace(r'\emptyset', '∅')
    text = text.replace(r'\forall', '∀')
    text = text.replace(r'\exists', '∃')
    text = text.replace(r'\nexists', '∄')
    text = text.replace(r'\sum', '∑')
    text = text.replace(r'\prod', '∏')
    text = text.replace(r'\lim', 'lim')
    text = text.replace(r'\to', '→')
    text = text.replace(r'\rightarrow', '→')
    text = text.replace(r'\leftarrow', '←')
    text = text.replace(r'\Rightarrow', '⇒')
    text = text.replace(r'\Leftrightarrow', '⇔')
    text = text.replace(r'\cdot', '·')
    text = text.replace(r'\ldots', '…')
    text = text.replace(r'\vdots', '⋮')
    text = text.replace(r'\cdots', '⋯')
    text = text.replace(r'\ddots', '⋱')
    greek_letters = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
        r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
        r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\rho': 'ρ',
        r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ', r'\phi': 'φ',
        r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Alpha': 'Α', r'\Beta': 'Β', r'\Gamma': 'Γ', r'\Delta': 'Δ',
        r'\Epsilon': 'Ε', r'\Zeta': 'Ζ', r'\Eta': 'Η', r'\Theta': 'Θ',
        r'\Iota': 'Ι', r'\Kappa': 'Κ', r'\Lambda': 'Λ', r'\Mu': 'Μ',
        r'\Nu': 'Ν', r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Rho': 'Ρ',
        r'\Sigma': 'Σ', r'\Tau': 'Τ', r'\Upsilon': 'Υ', r'\Phi': 'Φ',
        r'\Chi': 'Χ', r'\Psi': 'Ψ', r'\Omega': 'Ω'
    }
    for latex_cmd, symbol in greek_letters.items():
        text = text.replace(latex_cmd, symbol)
    text = re.sub(r'([a-zA-Z])_([0-9])', lambda m: f"{m.group(1)}{chr(0x2080 + int(m.group(2)))}", text)
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)
    return text


def escape_html(text: str) -> str:
    """Экранирует HTML символы в тексте"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('&', '&amp;').replace('<', '<').replace('>', '>')


def get_user_model(user_id: int) -> str:
    return user_models.get(user_id, DEFAULT_MODEL)


def get_model_name(model_id: str) -> str:
    for name, data in AVAILABLE_MODELS.items():
        if data["id"] == model_id:
            return name
    return model_id


def get_model_support_images(model_id: str) -> bool:
    for data in AVAILABLE_MODELS.values():
        if data["id"] == model_id:
            return data["image_support"]
    return False


def compress_image_high_quality(image_data: bytes, max_side: int = IMAGE_MAX_SIZE,
                                quality: int = IMAGE_QUALITY) -> bytes:
    try:
        img = Image.open(BytesIO(image_data))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        w, h = img.size
        max_dim = max(w, h)
        if max_dim > max_side:
            ratio = max_side / max_dim
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        logging.exception("compress_image_high_quality error")
        return image_data


def estimate_tokens_from_bytes(n_bytes: int) -> int:
    """Rough heuristic: tokens ≈ bytes / 3"""
    return max(1, int(n_bytes / 3))


async def try_delete_message(message: types.Message):
    try:
        await message.delete()
    except TelegramBadRequest as e:
        if "message to delete not found" in str(e):
            logging.debug("Message already deleted, skipping")
        else:
            logging.warning(f"Error deleting message: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error when deleting message")


async def extract_text_from_image(image_bytes: bytes, user_id: int) -> Optional[str]:
    """OCR с проверкой и списанием токенов"""
    # Проверка токенов
    if not db_use_tokens(user_id, TOKEN_COST_OCR):
        return None  # Недостаточно токенов

    try:
        jpeg_bytes = compress_image_high_quality(image_bytes, max_side=1024, quality=85)
        key = f"ocr_uploads/{uuid4().hex}.jpg"
        presigned_url = await upload_bytes_to_s3_and_get_presigned_url(jpeg_bytes, key, expires=S3_PRESIGNED_EXPIRATION)
        if not presigned_url:
            logging.error("Failed to upload image for OCR")
            return None
        if S3_AUTO_DELETE_AFTER > 0:
            await schedule_s3_delete(key, delay=S3_AUTO_DELETE_AFTER)
        content = [
            {"type": "text", "text": "Extract all text from this image. Return only the text content, nothing else."},
            {"type": "image_url", "image_url": {"url": presigned_url}}
        ]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OCR_MODEL,
            "messages": [{"role": "user", "content": content}]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                    json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OCR OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    return None
                if 'error' in data:
                    logging.error(f"OCR Error: {data['error']}")
                    return None
                if 'choices' not in data or not data['choices']:
                    logging.error(f"OCR Unexpected response format: {str(data)[:500]}")
                    return None
                extracted_text = data['choices'][0]['message']['content']
                logging.info(f"OCR extracted text: {extracted_text[:200]}...")
                return extracted_text
    except Exception as e:
        logging.exception("OCR processing error")
        return None


# ---------- S3 / Cloudflare R2 helpers ----------
_s3_client = None


def _make_s3_client_sync():
    global _s3_client
    if _s3_client is None:
        config = Config(signature_version='s3v4')
        params = {
            "aws_access_key_id": S3_ACCESS_KEY,
            "aws_secret_access_key": S3_SECRET_KEY,
            "config": config,
        }
        if S3_REGION:
            params["region_name"] = S3_REGION
        if S3_ENDPOINT_URL:
            params["endpoint_url"] = S3_ENDPOINT_URL
        _s3_client = boto3.client("s3", **params)
    return _s3_client


def _s3_put_object_sync(key: str, body: bytes, content_type: str):
    client = _make_s3_client_sync()
    return client.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType=content_type)


def _s3_generate_presigned_get_sync(key: str, expires_in: int):
    client = _make_s3_client_sync()
    return client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': key}, ExpiresIn=expires_in)


def _s3_delete_object_sync(key: str):
    client = _make_s3_client_sync()
    return client.delete_object(Bucket=S3_BUCKET, Key=key)


async def upload_bytes_to_s3_and_get_presigned_url(bytes_data: bytes, key: str,
                                                   expires: int = S3_PRESIGNED_EXPIRATION) -> Optional[str]:
    loop = asyncio.get_running_loop()
    try:
        # put object
        await loop.run_in_executor(executor, _s3_put_object_sync, key, bytes_data, "image/jpeg")
        # generate presigned url
        url = await loop.run_in_executor(executor, _s3_generate_presigned_get_sync, key, expires)
        return url
    except Exception:
        logging.exception("upload_bytes_to_s3_and_get_presigned_url error")
        return None


async def schedule_s3_delete(key: str, delay: int = S3_AUTO_DELETE_AFTER):
    if delay <= 0:
        return

    async def _del_later():
        try:
            await asyncio.sleep(delay)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, _s3_delete_object_sync, key)
            logging.info(f"S3: auto-deleted key={key}")
        except Exception:
            logging.exception("Error deleting S3 object in scheduled task")

    asyncio.create_task(_del_later())


# ---------- Commands & UI ----------
async def set_main_menu():
    cmds = [
        BotCommand(command="/start", description="Запустить бота"),
        BotCommand(command="/help", description="Помощь"),
        BotCommand(command="/models", description="Выбрать модель"),
        BotCommand(command="/currentmodel", description="Текущая модель"),
        BotCommand(command="/imagine", description="Создать изображение"),
        BotCommand(command="/profile", description="Мой профиль"),
        BotCommand(command="/buy_tokens", description="Докупить токенов")
    ]
    await bot.set_my_commands(cmds, scope=BotCommandScopeDefault())


@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    try:
        user_id = m.from_user.id
        referrer_id = None

        # Проверяем, есть ли параметр в команде /start
        if m.text and len(m.text.split()) > 1:
            ref_code_or_id = m.text.split()[1]
            # Пытаемся получить referrer_id (простой способ - если это ID)
            try:
                potential_referrer_id = int(ref_code_or_id)
                # Проверяем, существует ли такой пользователь и он не является самим собой
                if potential_referrer_id != user_id and potential_referrer_id > 0:
                    referrer_id = potential_referrer_id
            except ValueError:
                pass  # В данном примере не обрабатываем коды

        # Проверяем, существует ли пользователь в БД
        existing_user = db_get_user(user_id)
        is_new_user = existing_user is None

        welcome_text = "<b>🤖 AI Chat Bot</b>\nОтправьте текст или изображение."

        if is_new_user:
            # Пробуем создать пользователя
            if db_create_user(user_id, referrer_id):
                if referrer_id:
                    # Пользователь пришёл по ссылке
                    # Отправляем сообщение пригласившему (если он есть в user_models, значит, ботался)
                    try:
                        # Проверим, существует ли пригласивший в БД (на всякий случай)
                        if db_get_user(referrer_id):
                            await bot.send_message(referrer_id,
                                                   f"🎉 Поздравляем! По вашей ссылке присоединился новый пользователь (ID: {user_id}). Вы получили {REFERRAL_TOKENS} токенов!")
                        # Можно добавить логику, чтобы не спамить, если пользователь давно не активен
                    except Exception as e:
                        logging.warning(f"Не удалось уведомить реферера {referrer_id}: {e}")

                    welcome_text += f"\n\n👋 Добро пожаловать! Вы получили {STARTING_TOKENS} токенов за переход по реферальной ссылке!"
                # Инициализируем модель пользователя (если нужно)
                if user_id not in user_models:
                    user_models[user_id] = DEFAULT_MODEL
            else:
                # Ошибка создания пользователя
                welcome_text = "<b>🤖 AI Chat Bot</b>\n❌ Ошибка регистрации. Попробуйте позже."
        # else: Пользователь уже существует, просто приветствие

        await m.answer(welcome_text)
    except Exception as e:
        logging.exception("Ошибка в обработчике /start")
        await m.answer("❌ Не удалось обработать команду. Пожалуйста, попробуйте позже.")


@dp.message(Command("help"))
async def cmd_help(m: types.Message):
    try:
        help_text = (
            "<b>🛠️ Команды:</b>\n"
            "/start - Запустить бота\n"
            "/help - Помощь\n"
            "/models - Выбрать модель\n"
            "/currentmodel - Текущая модель\n"
            "/imagine - Создание изображения по запросу\n"
            "/profile - Мой профиль\n"
            "/buy_tokens - Покупка токенов за звёзды\n"
            "<b>Доступные модели:</b>\n"
        )
        for model_name, model_data in AVAILABLE_MODELS.items():
            help_text += f"• {escape_html(model_name)}\n"
        await m.answer(help_text)
    except Exception as e:
        logging.exception("Ошибка в обработчике /help")
        await m.answer("❌ Не удалось показать справку. Пожалуйста, попробуйте позже.")


@dp.message(Command("buy_tokens"))
async def cmd_buy_tokens(message: types.Message):
    try:
        user_id = message.from_user.id

        # Создаем клавиатуру с вариантами покупки
        builder = InlineKeyboardBuilder()
        builder.row(
            InlineKeyboardButton(text="100 токенов (10 Stars)", callback_data="buy_100"),
            InlineKeyboardButton(text="500 токенов (40 Stars)", callback_data="buy_500")
        )
        builder.row(
            InlineKeyboardButton(text="1000 токенов (70 Stars)", callback_data="buy_1000"),
            InlineKeyboardButton(text="Отмена", callback_data="cancel_purchase")
        )

        await message.answer(
            "🎁 <b>Выберите пакет токенов:</b>\n\n"
            "• 100 токенов - 10 Stars\n"
            "• 500 токенов - 40 Stars\n"
            "• 1000 токенов - 70 Stars\n\n"
            "<i>Telegram Stars - это внутренняя валюта Telegram для покупки цифровых товаров.</i>",
            reply_markup=builder.as_markup()
        )
    except Exception as e:
        logging.exception("Ошибка в обработчике /buy_tokens")
        await message.answer("❌ Не удалось показать варианты покупки. Попробуйте позже.")


@dp.callback_query(F.data.startswith("buy_"))
async def process_buy_callback(callback: types.CallbackQuery):
    try:
        user_id = callback.from_user.id
        pack_type = callback.data.split("_")[1]

        # Определяем параметры в зависимости от выбранного пакета
        if pack_type == "100":
            amount = 1  # 10 Stars в минимальных единицах (1 Star = 100)
            tokens = 100
            description = "Пакет из 100 токенов"
        elif pack_type == "500":
            amount = 40  # 40 Stars
            tokens = 500
            description = "Пакет из 500 токенов"
        elif pack_type == "1000":
            amount = 70  # 70 Stars
            tokens = 1000
            description = "Пакет из 1000 токенов"
        else:
            await callback.answer("Неизвестный пакет")
            return

        # Отправляем счет
        await callback.message.delete()
        await bot.send_invoice(
            chat_id=user_id,
            title="Покупка токенов",
            description=description,
            payload=f"tokens_{tokens}_{user_id}",
            currency="XTR",  # Валюта Telegram Stars
            prices=[LabeledPrice(label=f"{tokens} токенов", amount=amount)],
            start_parameter="buy_tokens",
            need_email=False,
            need_phone_number=False,
            need_shipping_address=False,
            is_flexible=False
        )
        await callback.answer()
    except Exception as e:
        logging.exception("Ошибка в обработчике выбора пакета токенов")
        await callback.answer("❌ Ошибка при создании счета. Попробуйте позже.", show_alert=True)

@dp.callback_query(F.data == "cancel_purchase")
async def cancel_purchase(callback: types.CallbackQuery):
    try:
        await callback.message.delete()
        await callback.answer("Покупка отменена")
    except Exception:
        await callback.answer()

@dp.pre_checkout_query()
async def process_pre_checkout(query: types.PreCheckoutQuery):
    try:
        # Всегда подтверждаем запрос, если он корректен
        await bot.answer_pre_checkout_query(query.id, ok=True)
    except Exception as e:
        logging.exception("Ошибка в обработчике предварительной проверки платежа")
        await bot.answer_pre_checkout_query(query.id, ok=False, error_message="Произошла ошибка при обработке платежа")


@dp.message(F.content_type == ContentType.SUCCESSFUL_PAYMENT)
async def process_successful_payment(message: types.Message):
    try:
        user_id = message.from_user.id
        payment_info = message.successful_payment

        # Извлекаем количество токенов из payload
        payload_parts = payment_info.invoice_payload.split("_")
        if len(payload_parts) < 3 or payload_parts[0] != "tokens":
            logging.error(f"Неверный формат payload: {payment_info.invoice_payload}")
            return

        tokens = int(payload_parts[1])
        target_user_id = int(payload_parts[2])

        # Проверяем, что платеж предназначен текущему пользователю
        if target_user_id != user_id:
            logging.error(f"Несоответствие user_id: {user_id} != {target_user_id}")
            return

        # Начисляем токены пользователю
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE users SET tokens = tokens + ? WHERE user_id = ?',
            (tokens, user_id)
        )
        conn.commit()
        conn.close()

        # Отправляем подтверждение
        await message.answer(
            f"✅ <b>Оплата прошла успешно!</b>\n"
            f"На ваш счет зачислено <b>{tokens} токенов</b>.\n"
            f"Теперь у вас <b>{db_get_user_stats(user_id)['tokens']} токенов</b>."
        )
    except Exception as e:
        logging.exception("Ошибка в обработчике успешного платежа")
        await message.answer("❌ Произошла ошибка при обработке платежа. Обратитесь к администратору.")
@dp.message(Command("imagine"))
async def cmd_imagine(m: types.Message, state: FSMContext):
    try:
        await state.set_state(ImageGenState.waiting_for_prompt)
        await m.answer(
            "🎨 Отправьте текстовый запрос для генерации изображения. Чтобы выйти из режима генерации, используйте /cancel.")
    except Exception as e:
        logging.exception("Ошибка в обработчике /imagine")
        await m.answer("❌ Не удалось активировать режим генерации изображений. Пожалуйста, попробуйте позже.")


@dp.message(StateFilter(ImageGenState.waiting_for_prompt), F.text.startswith('/'))
async def handle_any_command_during_imagine(m: types.Message, state: FSMContext):
    try:
        await state.clear()

        if m.text.startswith("/cancel"):
            await m.answer("❌ Режим генерации изображений отменен.")
            return

        await m.answer("❌ Режим генерации изображений отменен.")

        if m.text.startswith("/start"):
            await cmd_start(m)
        elif m.text.startswith("/help"):
            await cmd_help(m)
        elif m.text.startswith("/models"):
            await list_models(m)
        elif m.text.startswith("/currentmodel"):
            await current_model(m)
        elif m.text.startswith("/profile"):
            await cmd_profile(m)
        elif m.text.startswith("/imagine"):
            await cmd_imagine(m, state)
    except Exception as e:
        logging.exception("Ошибка в обработчике команды во время ожидания промпта")
        await m.answer("❌ Не удалось обработать команду. Пожалуйста, попробуйте позже.")


@dp.message(StateFilter(ImageGenState.waiting_for_prompt))
async def handle_image_prompt(m: types.Message, state: FSMContext):
    try:
        translator = Translator()
        user_id = m.from_user.id
        translation = await translator.translate(m.text.strip())
        prompt = translation.text
        print(prompt)
        if not prompt:
            return await m.answer("⚠️ Промпт не может быть пустым.")

        if not db_use_tokens(user_id, TOKEN_COST_IMAGE_GEN):
            stats = db_get_user_stats(user_id)
            await state.clear()  # Выходим из состояния при ошибке
            return await m.answer(
                f"⚠️ Недостаточно токенов для генерации изображения. У вас {stats['tokens']} токенов, требуется {TOKEN_COST_IMAGE_GEN}.")

        await bot.send_chat_action(m.chat.id, "upload_photo")
        status = await m.answer("<i>🎨 Генерирую изображение через Together AI (FLUX.1-schnell-Free)...</i>")

        try:
            headers = {
                "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "black-forest-labs/FLUX.1-schnell-Free",
                "prompt": prompt,
                "steps": 4,
                "width": 1024,
                "height": 1024
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(TOGETHER_AI_API_URL, headers=headers, json=payload) as resp:
                    logging.info(f"Together AI Response Status: {resp.status}")
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                        except Exception as e:
                            text_error = await resp.text()
                            logging.error(f"Together AI returned non-json: status={resp.status} text={text_error[:500]}")
                            await status.delete()
                            await state.clear()
                            return await m.answer("❌ Не удалось обработать ответ от сервера генерации.")
                        logging.info(f"Together AI JSON Response: {data}")
                        if 'data' not in data or not data['data'] or not isinstance(data['data'], list):
                            await status.delete()
                            await state.clear()
                            return await m.answer("❌ Неожиданный формат ответа от сервера генерации.")
                        image_info = data['data'][0]
                        image_data_bytes = None
                        if 'b64_json' in image_info and image_info['b64_json']:
                            try:
                                image_data_bytes = base64.b64decode(image_info['b64_json'])
                            except Exception as e:
                                logging.error(f"Error decoding base64 image: {e}")
                                await status.delete()
                                await state.clear()
                                return await m.answer("❌ Ошибка декодирования изображения.")
                        elif 'url' in image_info and image_info['url']:
                            try:
                                async with session.get(image_info['url']) as img_resp:
                                    if img_resp.status == 200:
                                        image_data_bytes = await img_resp.read()
                                    else:
                                        await status.delete()
                                        await state.clear()  # Выходим из состояния при ошибке
                                        return await m.answer("❌ Не удалось скачать изображение.")
                            except Exception as e:
                                logging.error(f"Error downloading image from URL: {e}")
                                await status.delete()
                                await state.clear()  # Выходим из состояния при ошибке
                                return await m.answer("❌ Ошибка скачивания изображения.")
                        else:
                            await status.delete()
                            await state.clear()  # Выходим из состояния при ошибке
                            return await m.answer("❌ Ответ не содержит данных изображения.")
                        if image_data_bytes:
                            await status.delete()
                            await m.answer_photo(
                                BufferedInputFile(image_data_bytes, filename="together_ai_image.jpg"),
                                caption=f"🖼️FLUX.1\n <b>Промпт:</b> {escape_html(m.text.strip())}"
                            )
                            # Оставляем состояние активным для продолжения генерации
                        else:
                            await status.delete()
                            await state.clear()  # Выходим из состояния при ошибке
                            await m.answer("❌ Не удалось получить изображение.")
                    else:
                        text_error = await resp.text()
                        logging.error(f"Together AI Error {resp.status}: {text_error}")
                        await status.delete()
                        await state.clear()  # Выходим из состояния при ошибке
                        await m.answer("❌ Ошибка генерации изображения. Пожалуйста попробуйте позже.")
        except Exception as e:
            logging.exception("Together AI image generation error")
            try:
                await status.delete()
            except:
                pass
            await state.clear()  # Выходим из состояния при ошибке
            await m.answer("❌ Ошибка генерации изображения. Попробуйте позже.")
    except Exception as e:
        logging.exception("Ошибка в обработчике промпта для изображения")
        await m.answer("❌ Не удалось обработать запрос на генерацию изображения. Пожалуйста, попробуйте позже.")
        await state.clear()  # Важно: сбрасываем состояние


@dp.message(Command("cancel"))
async def cmd_cancel(m: types.Message):
    pass


@dp.message(Command("models"))
async def list_models(m: types.Message):
    try:
        await show_models_keyboard(m, m.from_user.id)
    except Exception as e:
        logging.exception("Ошибка в обработчике /models")
        await m.answer("❌ Не удалось показать список моделей. Пожалуйста, попробуйте позже.")


async def show_models_keyboard(message: types.Message, user_id: int, edit: bool = False):
    try:
        builder = InlineKeyboardBuilder()
        current_model_id = get_user_model(user_id)
        new_text = f"🛠️ <b>Выберите модель:</b>\nТекущая: {escape_html(get_model_name(current_model_id))}\n<i>Нажмите на модель</i>"
        for model_name, model_data in AVAILABLE_MODELS.items():
            prefix = "✅ " if model_data["id"] == current_model_id else ""
            builder.button(text=f"{prefix}{model_name}", callback_data=f"model_{model_data['id']}")
        builder.button(text="❌ Закрыть", callback_data="close_menu")
        builder.adjust(2, 2, 2)
        new_markup = builder.as_markup()
        if edit:
            try:
                current_text = message.html_text
                current_markup = message.reply_markup
                if new_text != current_text or str(new_markup) != str(current_markup):
                    await message.edit_text(new_text, reply_markup=new_markup)
                else:
                    logging.debug("Message content not changed, skipping edit")
            except Exception as e:
                logging.error(f"Error editing message: {e}")
        else:
            await message.answer(new_text, reply_markup=new_markup)
    except Exception as e:
        logging.exception("Ошибка в show_models_keyboard")
        if not edit:
            await message.answer("❌ Не удалось показать модели. Пожалуйста, попробуйте позже.")


@dp.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
    try:
        model_id = callback.data.split("_", 1)[1]
        user_id = callback.from_user.id
        model_exists = any(model_data["id"] == model_id for model_data in AVAILABLE_MODELS.values())
        if model_exists:
            user_models[user_id] = model_id
            await show_models_keyboard(callback.message, user_id, edit=True)
            model_name = escape_html(get_model_name(model_id))
            await callback.answer(f"Модель изменена на {model_name}")
        else:
            await callback.answer("⚠️ Неизвестная модель", show_alert=True)
    except Exception as e:
        logging.exception("Ошибка в обработчике выбора модели")
        await callback.answer("❌ Не удалось изменить модель. Пожалуйста, попробуйте позже.", show_alert=True)


@dp.callback_query(F.data == "close_menu")
async def close_menu_callback(callback: types.CallbackQuery):
    try:
        await callback.message.delete()
    except Exception:
        pass
    await callback.answer("Меню закрыто")


@dp.message(Command("currentmodel"))
async def current_model(m: types.Message):
    try:
        user_id = m.from_user.id
        model_id = get_user_model(user_id)
        model_name = escape_html(get_model_name(model_id))

        stats = db_get_user_stats(user_id)
        tokens = stats.get("tokens", 0)

        await m.answer(
            f"🔧 <b>Текущая модель:</b> {model_name}\n🎁 <b>Токены:</b> {tokens}")
    except Exception as e:
        logging.exception("Ошибка в обработчике /currentmodel")
        await m.answer("❌ Не удалось показать текущую модель. Пожалуйста, попробуйте позже.")


@dp.message(Command("profile"))
async def cmd_profile(m: types.Message):
    try:
        user_id = m.from_user.id
        # Убедимся, что пользователь существует в БД
        existing_user = db_get_user(user_id)
        if existing_user is None:
            # Регистрируем пользователя без реферера
            db_create_user(user_id)

        # Получаем статистику
        stats = db_get_user_stats(user_id)
        referred_count = stats.get("referral_count", 0)
        tokens = stats.get("tokens", 0)

        # Реферальная ссылка
        bot_info = await bot.get_me()
        bot_username = bot_info.username
        if not bot_username:
            await m.answer("⚠️ Бот должен иметь имя пользователя (@username) для создания реферальных ссылок.")
            return

        referral_link = f"https://t.me/{bot_username}?start={user_id}"

        response_text = (
            f"<b>👤 Мой профиль</b>\n\n"
            f"<b>🔗 Реферальная ссылка:</b>\n"
            f"<code>{referral_link}</code>\n\n"
            f"<b>📊 Статистика:</b>\n"
            f"Приглашено друзей: {referred_count}\n"
            f"Токены: {tokens}"
        )

        await m.answer(response_text, parse_mode=ParseMode.HTML)
    except Exception as e:
        logging.exception("Ошибка в обработчике /profile")
        await m.answer("❌ Не удалось показать профиль. Пожалуйста, попробуйте позже.")


@dp.message(Command("upload_test"))
async def cmd_upload_test(m: types.Message):
    try:
        user_id = m.from_user.id
        # Проверка токенов
        if not db_use_tokens(user_id, TOKEN_COST_OCR):
            stats = db_get_user_stats(user_id)
            return await m.answer(
                f"⚠️ Недостаточно токенов для OCR. У вас {stats['tokens']} токенов, требуется {TOKEN_COST_OCR}.")

        test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1024px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        model_id = get_user_model(user_id)
        if not get_model_support_images(model_id):
            return await m.answer(
                "Текущая модель не поддерживает изображения. Выберите модель с поддержкой изображений (/models).")
        status = await m.answer("<i>Тестовый запрос: отправляю публичный URL...</i>")
        content = [{"type": "text", "text": "Опиши это изображение (тестовый публичный URL)"},
                   {"type": "image_url", "image_url": {"url": test_url}}]
        payload = {"model": model_id, "messages": [{"role": "user", "content": content}]}
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                    json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    await status.delete()
                    return await m.answer("❌ Ошибка обработки ответа от сервера.")
        await status.delete()
        await m.answer(f"OpenRouter response status {resp.status}. Usage: {data.get('usage')}\nPreview: {str(data)[:1000]}")
    except Exception as e:
        logging.exception("Ошибка в обработчике /upload_test")
        await m.answer("❌ Не удалось выполнить тест загрузки. Пожалуйста, попробуйте позже.")


# ---------- Image processing ----------
async def process_image_message(message: types.Message, image_bytes: bytes, caption: str = ""):
    user_id = message.from_user.id
    status_msg = await message.answer("<i>🤖 Обрабатываю изображение...</i>")
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(user_id)
        target_model_supports_images = get_model_support_images(model_id)
        if target_model_supports_images:
            await process_image_directly(message, image_bytes, caption, model_id, status_msg, user_id)
        else:
            await process_image_with_ocr(message, image_bytes, caption, model_id, status_msg, user_id)
    except Exception:
        logging.exception("process_image_message error")
        try:
            await try_delete_message(status_msg)
        except Exception:
            pass
        await message.answer("❌ Не удалось обработать изображение. Попробуйте позже.")


async def process_image_with_ocr(message: types.Message, image_bytes: bytes, caption: str, model_id: str, status_msg,
                                 user_id: int):
    try:
        try:
            await status_msg.edit_text("<i>🔍 Распознаю текст на изображении...</i>")
        except Exception:
            pass
        extracted_text = await extract_text_from_image(image_bytes, user_id)  # Передаем user_id
        if not extracted_text:
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            stats = db_get_user_stats(user_id)
            return await message.answer(
                f"⚠️ Не удалось распознать текст на изображении. Возможно, недостаточно токенов. У вас {stats['tokens']} токенов.")
        prompt = f"Изображение содержит следующий текст:\n{extracted_text}\n"
        if caption:
            prompt += f"Пользователь добавил: {caption}\n"
        prompt += "Пожалуйста, ответь на запрос пользователя, используя информацию из текста выше."
        try:
            await status_msg.edit_text("<i>🤖 Обрабатываю извлеченный текст...</i>")
        except Exception:
            pass
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model_id, "messages": [{"role": "user", "content": prompt}]}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                    json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    try:
                        await try_delete_message(status_msg)
                    except Exception:
                        pass
                    return await message.answer("❌ Ошибка обработки ответа от сервера.")
        logging.info(f"OpenRouter response: {data}")
        if 'error' in data:
            error_msg = data['error']
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            return await message.answer("❌ Ошибка обработки изображения.")
        if 'choices' not in data or not data['choices']:
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            return await message.answer("❌ Неожиданный формат ответа от сервера.")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        last_user_message[user_id]=caption
        last_bot_responses[user_id]=ai_response
        try:
            await try_delete_message(status_msg)
        except Exception:
            pass
        model_name = escape_html(get_model_name(model_id))
        response_text = f"🧠 <b>{model_name}</b>:\n{formatted}"
        if len(response_text) > 4096:
            for i in range(0, len(response_text), 4096):
                await message.answer(response_text[i:i + 4096])
        else:
            await message.answer(response_text)
    except Exception as e:
        logging.exception("process_image_with_ocr error")
        try:
            await try_delete_message(status_msg)
        except Exception:
            pass
        await message.answer("❌ Ошибка при обработке изображения через OCR. Попробуйте позже.")


async def process_image_directly(message: types.Message, image_bytes: bytes, caption: str, model_id: str, status_msg,
                                 user_id: int):
    # Проверка токенов
    if not db_use_tokens(user_id, TOKEN_COST_OCR):
        stats = db_get_user_stats(user_id)
        await try_delete_message(status_msg)
        return await message.answer(
            f"⚠️ Недостаточно токенов для обработки изображения. У вас {stats['tokens']} токенов, требуется {TOKEN_COST_OCR}.")

    try:
        jpeg_bytes = compress_image_high_quality(image_bytes)
        compressed_n = len(jpeg_bytes)
        logging.info(
            f"Original bytes: {len(image_bytes)}; JPEG bytes: {compressed_n}; est tokens if inlined: {estimate_tokens_from_bytes(compressed_n)}")
        key = f"telegram_uploads/{uuid4().hex}.jpg"
        presigned_url = await upload_bytes_to_s3_and_get_presigned_url(jpeg_bytes, key, expires=S3_PRESIGNED_EXPIRATION)
        if not presigned_url:
            await try_delete_message(status_msg)
            logging.error("Failed to upload to S3 / generate presigned URL")
            return await message.answer("❌ Не удалось загрузить изображение в облако. Повторите позже.")
        logging.info(f"Uploaded to S3 key={key} presigned_url={presigned_url}")
        if S3_AUTO_DELETE_AFTER > 0:
            await schedule_s3_delete(key, delay=S3_AUTO_DELETE_AFTER)
        content = []
        content.append({"type": "text", "text": caption or "Опиши это изображение"})
        content.append({"type": "image_url", "image_url": {"url": presigned_url}})
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model_id, "messages": [{"role": "user", "content": content}]}
        logging.info(f"Sending to OpenRouter model={model_id} payload=image_url (presigned S3)")
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                    json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    await try_delete_message(status_msg)
                    return await message.answer("❌ Ошибка обработки ответа от сервера.")
        logging.info(f"OpenRouter response: {data}")
        await try_delete_message(status_msg)
        if 'error' in data:
            error_msg = data['error']
            return await message.answer("❌ Ошибка обработки изображения.")
        if 'choices' not in data or not data['choices']:
            error_info = str(data)[:500] if data else "Empty response"
            return await message.answer("❌ Неожиданный формат ответа от сервера.")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        await try_delete_message(status_msg)
        model_name = escape_html(get_model_name(model_id))
        response_text = f"🧠 <b>{model_name}</b>:\n{formatted}"
        if len(response_text) > 4096:
            for i in range(0, len(response_text), 4096):
                await message.answer(response_text[i:i + 4096])
        else:
            await message.answer(response_text)
    except Exception as e:
        logging.exception("process_image_directly error")
        try:
            await try_delete_message(status_msg)
        except Exception:
            pass
        await message.answer("❌ Ошибка при обработке изображения. Попробуйте позже.")


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    try:
        photo = message.photo[-1]
        downloaded = await bot.download_file(photo.file_path)
        image_bytes = downloaded.read()
    except Exception:
        try:
            f = await bot.get_file(message.photo[-1].file_id)
            downloaded = await bot.download_file(f.file_path)
            image_bytes = downloaded.read()
        except Exception:
            logging.exception("Failed to download photo")
            return await message.answer("❌ Не удалось скачать изображение.")
    await process_image_message(message, image_bytes, message.caption or "")


# ---------- Text messages ----------
@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return await message.answer("Отправьте текстовое сообщение.")

    user_id = message.from_user.id
    user_text = message.text.strip()

    # Проверка токенов
    if not db_use_tokens(user_id, TOKEN_COST_TEXT):
        stats = db_get_user_stats(user_id)
        return await message.answer(
            f"⚠️ Недостаточно токенов для текстового запроса. У вас {stats['tokens']} токенов, требуется {TOKEN_COST_TEXT}.")

    # --- Получаем последний ответ бота ---
    last_response = last_bot_responses.get(user_id)
    last_message = last_user_message.get(user_id)
    print(last_message, last_response)
    # --- Формируем список сообщений для OpenRouter ---
    messages = [
        {"role": "user", "content": user_text}
    ]
    # --- Добавляем предыдущий ответ бота как контекст ---
    if last_response:
        messages.insert(0, {
            "role": "assistant",
            "content": last_response
        })
    if last_message:
        messages.insert(0, {
            "role": "user", "content": last_message
        })
    status = await message.answer("<i>🤖 Обрабатываю...</i>")
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(user_id)
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_id,
            "messages": messages  # Здесь теперь история!
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
            ) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    await try_delete_message(status)
                    return await message.answer("❌ Ошибка обработки ответа от сервера.")
        logging.info(f"OpenRouter response: {data}")
        await try_delete_message(status)
        if 'error' in data:
            error_msg = data['error']
            return await message.answer("❌ Ошибка обработки запроса.")
        if 'choices' not in data or not data['choices']:
            error_info = str(data)[:500] if data else "Empty response"
            return await message.answer("❌ Неожиданный формат ответа от сервера.")
        logging.info(f"OpenRouter usage: {data.get('usage')}")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        # --- Сохраняем новый ответ ---
        last_bot_responses[user_id] = ai_response
        last_user_message[user_id] = user_text
        response = f"🧠 <b>{escape_html(get_model_name(model_id))}</b>:\n{formatted}"
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                await message.answer(response[i:i + 4096], parse_mode=ParseMode.HTML)
        else:
            await message.answer(response, parse_mode=ParseMode.HTML)
    except Exception:
        logging.exception("handle_message error")
        try:
            await try_delete_message(status)
        except Exception:
            pass
        await message.answer("❌ Ошибка при обработке запроса. Попробуйте позже.")


# ---------- Run ----------
async def main():
    # --- Инициализация БД ---
    init_db()
    # --- Конец инициализации БД ---
    await set_main_menu()
    logging.info("✅ Бот запущен")
    logging.info(f"S3_ENABLED={S3_ENABLED}, S3_BUCKET={S3_BUCKET}, S3_ENDPOINT_URL={S3_ENDPOINT_URL}")
    logging.info(f"IMAGE_MAX_SIZE={IMAGE_MAX_SIZE}, IMAGE_QUALITY={IMAGE_QUALITY}")
    logging.info(f"S3_PRESIGNED_EXPIRATION={S3_PRESIGNED_EXPIRATION}, S3_AUTO_DELETE_AFTER={S3_AUTO_DELETE_AFTER}")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")