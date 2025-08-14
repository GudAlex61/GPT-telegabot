#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import aiohttp
import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeDefault
from aiogram.exceptions import TelegramBadRequest
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

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
# Models that accept image_url (IDs)
MODELS_WITH_IMAGE_URL = {
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-exp:free"
}

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


# ---------- Utilities ----------
def convert_markdown_to_html(text: str) -> str:
    """
    Конвертирует markdown от нейросетей в HTML с правильным экранированием
    """
    if not text:
        return ""

    import re

    math_blocks = []
    math_inlines = []
    code_blocks = []

    def save_math_block(match):
        placeholder = f"{{MATH_BLOCK_{len(math_blocks)}}}"
        math_content = match.group(1)
        math_content = replace_math_symbols(math_content)
        math_blocks.append(math_content)
        return placeholder

    text = re.sub(r'\\\[(.*?)\\\]', save_math_block, text, flags=re.DOTALL)

    def save_math_inline(match):
        placeholder = f"{{MATH_INLINE_{len(math_inlines)}}}"
        math_content = match.group(1)
        math_content = replace_math_symbols(math_content)
        math_inlines.append(math_content)
        return placeholder

    text = re.sub(r'\\\((.*?)\\\)', save_math_inline, text, flags=re.DOTALL)

    def save_code_block(match):
        placeholder = f"{{CODE_BLOCK_{len(code_blocks)}}}"
        language = match.group(1) or ""
        code_content = match.group(2)
        code_blocks.append((language, code_content))
        return placeholder

    # Обрабатываем блоки кода с указанием языка: ```language\ncontent```
    text = re.sub(r'```(\w+)\n(.*?)```', save_code_block, text, flags=re.DOTALL)
    # Обрабатываем блоки кода без указания языка: ```\ncontent```
    text = re.sub(r'```\n(.*?)```', save_code_block, text, flags=re.DOTALL)

    text = text.replace('&', '&amp;').replace('<', '<').replace('>', '>')

    for i, math_block in enumerate(math_blocks):
        text = text.replace(f'{{MATH_BLOCK_{i}}}', f'<pre><code>{math_block}</code></pre>')

    for i, math_inline in enumerate(math_inlines):
        text = text.replace(f'{{MATH_INLINE_{i}}}', f'<code>{math_inline}</code>')

    # Восстанавливаем блоки кода
    for i, (language, code_content) in enumerate(code_blocks):
        if language:
            # Если указан язык, добавляем его как класс
            text = text.replace(f'{{CODE_BLOCK_{i}}}', f'<pre><code class="language-{language}">{code_content}</code></pre>')
        else:
            text = text.replace(f'{{CODE_BLOCK_{i}}}', f'<pre><code>{code_content}</code></pre>')

    # Обработка заголовков
    text = re.sub(r'^######\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#####\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^####\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.*?)$', r'<b>\1</b>', text, flags=re.MULTILINE)

    # Базовое форматирование
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Обычные блоки кода без языка (если остались)
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
    """
    Compress/resize image into JPEG for upload (high quality).
    """
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

async def extract_text_from_image(image_bytes: bytes) -> Optional[str]:
    """
    Использует OCR модель для извлечения текста из изображения
    """
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
    """
    Uploads bytes_data to S3 (key) and returns presigned GET URL (expires seconds).
    Uses run_in_executor to avoid blocking the event loop.
    """
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
    """Schedule deletion of key after delay seconds (best-effort)."""
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
        BotCommand(command="/imagine", description="Создать изображение")
    ]
    await bot.set_my_commands(cmds, scope=BotCommandScopeDefault())


@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    await m.answer(
        "<b>🤖 AI Chat Bot</b>\n\n"
        "Отправьте текст или изображение."
    )


@dp.message(Command("help"))
async def cmd_help(m: types.Message):
    help_text = (
        "<b>🛠️ Команды:</b>\n"
        "/start - Запустить бота\n"
        "/help - Помощь\n"
        "/models - Выбрать модель\n"
        "/currentmodel - Текущая модель\n"
        "/imagine - Создание изображения по запросу\n\n"
        "<b>Доступные модели:</b>\n"
    )
    for model_name, model_data in AVAILABLE_MODELS.items():
        help_text += f"• {escape_html(model_name)}\n"
    await m.answer(help_text)

@dp.message(Command("imagine"))
async def cmd_imagine(m: types.Message):
    if not m.text or len(m.text.strip()) <= len("/imagine"):
        return await m.answer(
            "🎨 Введите промпт после команды.\n"
            "Пример: <code>/imagine a cyberpunk cat, 4k detailed</code>",
            parse_mode=ParseMode.HTML
        )

    prompt = m.text[len("/imagine"):].strip()
    if not prompt:
        return await m.answer("⚠️ Промпт не может быть пустым.")

    await bot.send_chat_action(m.chat.id, "upload_photo")
    status = await m.answer("<i>🎨 Генерирую изображение через SDXL...</i>")

    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            # SDXL поддерживает параметры
            # "parameters": {
            #     "negative_prompt": "blurry, bad quality, ugly",
            #     "guidance_scale": 7.5,
            #     "num_inference_steps": 30
            # }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(HF_API_URL, headers=headers, json=payload) as resp:
                logging.info(f"SDXL Response Status: {resp.status}")
                if resp.status == 200:
                    image_data = await resp.read()
                    await status.delete()
                    await m.answer_photo(
                        BufferedInputFile(image_data, filename="sdxl_image.jpg"),
                        caption=f"🖼️ SDXL: {escape_html(prompt)}"
                    )
                elif resp.status == 503:
                    try:
                        data = await resp.json()
                        est_time = data.get("estimated_time", 20)
                        await status.edit_text(f"<i>⏳ SDXL загружается... (~{int(est_time)} секунд)</i>")
                    except Exception:
                        await status.edit_text("<i>⏳ SDXL загружается...</i>")
                else:
                    text_error = await resp.text()
                    logging.error(f"SDXL Error {resp.status}: {text_error}")
                    await status.delete()
                    await m.answer(
                        f"⚠️ Ошибка генерации ({resp.status}):\n"
                        f"<pre>{html_escape(text_error[:600])}</pre>",
                        parse_mode=ParseMode.HTML
                    )
    except Exception as e:
        logging.exception("SDXL image generation error")
        try:
            await status.delete()
        except:
            pass
        await m.answer(f"🚫 Ошибка: {str(e)}")

@dp.message(Command("models"))
async def list_models(m: types.Message):
    await show_models_keyboard(m, m.from_user.id)


async def show_models_keyboard(message: types.Message, user_id: int, edit: bool = False):
    builder = InlineKeyboardBuilder()
    current_model_id = get_user_model(user_id)

    # Собираем новый текст и клавиатуру
    new_text = f"🛠️ <b>Выберите модель:</b>\nТекущая: {escape_html(get_model_name(current_model_id))}\n\n<i>Нажмите на модель</i>"

    for model_name, model_data in AVAILABLE_MODELS.items():
        prefix = "✅ " if model_data["id"] == current_model_id else ""
        camera_icon = " 📷" if model_data["image_support"] else ""
        builder.button(text=f"{prefix}{model_name}{camera_icon}", callback_data=f"model_{model_data['id']}")

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


@dp.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
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


@dp.callback_query(F.data == "close_menu")
async def close_menu_callback(callback: types.CallbackQuery):
    try:
        await callback.message.delete()
    except Exception:
        pass
    await callback.answer("Меню закрыто")


@dp.message(Command("currentmodel"))
async def current_model(m: types.Message):
    model_id = get_user_model(m.from_user.id)
    model_name = escape_html(get_model_name(model_id))
    image_support = "Да" if get_model_support_images(model_id) else "Нет"
    await m.answer(f"🔧 <b>Текущая модель:</b> {model_name}\n📸 <b>Поддержка изображений:</b> {image_support}")


@dp.message(Command("upload_test"))
async def cmd_upload_test(m: types.Message):
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1024px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg  "
    model_id = get_user_model(m.from_user.id)
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
                return await m.answer(f"Ошибка OpenRouter: status={resp.status} text={text[:400]}")
    await status.delete()
    await m.answer(f"OpenRouter response status {resp.status}. Usage: {data.get('usage')}\nPreview: {str(data)[:1000]}")


# ---------- Image processing ----------
async def process_image_message(message: types.Message, image_bytes: bytes, caption: str = ""):
    status_msg = await message.answer("<i>🤖 Обрабатываю изображение...</i>")
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(message.from_user.id)

        target_model_supports_images = get_model_support_images(model_id)

        if target_model_supports_images:
            await process_image_directly(message, image_bytes, caption, model_id, status_msg)
        else:
            await process_image_with_ocr(message, image_bytes, caption, model_id, status_msg)

    except Exception:
        logging.exception("process_image_message error")
        try:
            await try_delete_message(status_msg)
        except Exception:
            pass
        await message.answer("🚫 Ошибка при обработке изображения")

async def process_image_with_ocr(message: types.Message, image_bytes: bytes, caption: str, model_id: str, status_msg):
    try:
        try:
            await status_msg.edit_text("<i>🔍 Распознаю текст на изображении...</i>")
        except Exception:
            pass

        extracted_text = await extract_text_from_image(image_bytes)

        if not extracted_text:
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            return await message.answer("⚠️ Не удалось распознать текст на изображении.")

        prompt = f"Изображение содержит следующий текст:\n\n{extracted_text}\n\n"
        if caption:
            prompt += f"Пользователь добавил: {caption}\n\n"
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
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status}): {text[:400]}")

        logging.info(f"OpenRouter response: {data}")

        if 'error' in data:
            error_msg = data['error']
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            return await message.answer(f"⚠️ Ошибка OpenRouter: {error_msg}")

        if 'choices' not in data or not data['choices']:
            try:
                await try_delete_message(status_msg)
            except Exception:
                pass
            return await message.answer(f"⚠️ Неожиданный формат ответа от OpenRouter: {str(data)[:500]}")

        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)

        try:
            await try_delete_message(status_msg)
        except Exception:
            pass

        model_name = escape_html(get_model_name(model_id))
        response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted}"

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
        await message.answer(f"🚫 Ошибка при обработке изображения через OCR: {str(e)}")


async def process_image_directly(message: types.Message, image_bytes: bytes, caption: str, model_id: str, status_msg):
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
            return await message.answer(
                "Не удалось загрузить изображение в облако (S3). Повторите позже или проверьте настройки.")

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
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status}): {text[:400]}")

        logging.info(f"OpenRouter response: {data}")
        await try_delete_message(status_msg)

        if 'error' in data:
            error_msg = data['error']
            return await message.answer(f"⚠️ Ошибка OpenRouter: {error_msg}")

        if 'choices' not in data or not data['choices']:
            error_info = str(data)[:500] if data else "Empty response"
            return await message.answer(f"⚠️ Неожиданный формат ответа от OpenRouter: {error_info}")

        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        await try_delete_message(status_msg)
        model_name = escape_html(get_model_name(model_id))
        response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted}"
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
        await message.answer(f"🚫 Ошибка при обработке изображения: {str(e)}")


async def process_image_with_ocr(message: types.Message, image_bytes: bytes, caption: str, model_id: str, status_msg):
    try:
        await status_msg.edit_text("<i>🔍 Распознаю текст на изображении...</i>")

        extracted_text = await extract_text_from_image(image_bytes)

        if not extracted_text:
            await try_delete_message(status_msg)
            return await message.answer("⚠️ Не удалось распознать текст на изображении.")

        prompt = f"Изображение содержит следующий текст:\n\n{extracted_text}\n\n"
        if caption:
            prompt += f"Пользователь добавил: {caption}\n\n"
        prompt += "Пожалуйста, ответь на запрос пользователя, используя информацию из текста выше."

        await status_msg.edit_text("<i>🤖 Обрабатываю извлеченный текст...</i>")

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
                    await try_delete_message(status_msg)
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status}): {text[:400]}")

        logging.info(f"OpenRouter response: {data}")
        await try_delete_message(status_msg)

        if 'error' in data:
            error_msg = data['error']
            return await message.answer(f"⚠️ Ошибка OpenRouter: {error_msg}")

        if 'choices' not in data or not data['choices']:
            error_info = str(data)[:500] if data else "Empty response"
            return await message.answer(f"⚠️ Неожиданный формат ответа от OpenRouter: {error_info}")

        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        await try_delete_message(status_msg)

        model_name = escape_html(get_model_name(model_id))
        response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted}"

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
        await message.answer(f"🚫 Ошибка при обработке изображения через OCR: {str(e)}")


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
            return await message.answer("⚠️ Не удалось скачать изображение.")
    await process_image_message(message, image_bytes, message.caption or "")


# ---------- Text messages ----------
@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return await message.answer("Отправьте текстовое сообщение.")
    status = await message.answer("<i>🤖 Обрабатываю...</i>")
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(message.from_user.id)
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": model_id, "messages": [{"role": "user", "content": message.text}]}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers,
                                    json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    await status.delete()
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status}): {text[:400]}")

        logging.info(f"OpenRouter response: {data}")
        await status.delete()

        if 'error' in data:
            error_msg = data['error']
            return await message.answer(f"⚠️ Ошибка OpenRouter: {error_msg}")

        if 'choices' not in data or not data['choices']:
            error_info = str(data)[:500] if data else "Empty response"
            return await message.answer(f"⚠️ Неожиданный формат ответа от OpenRouter: {error_info}")

        logging.info(f"OpenRouter usage: {data.get('usage')}")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        response = f"🧠 <b>{escape_html(get_model_name(model_id))}</b>:\n\n{formatted}"
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                await message.answer(response[i:i + 4096])
        else:
            await message.answer(response)
    except Exception:
        logging.exception("handle_message error")
        try:
            await status.delete()
        except Exception:
            pass
        await message.answer("🚫 Ошибка при обработке запроса")

# ---------- Run ----------
async def main():
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