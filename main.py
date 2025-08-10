#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Telegram bot with Cloudflare R2 (S3) integration.
Uploads compressed JPEG to S3, generates presigned GET URL and sends that url
to multimodal models that support image_url. No inline/base64 fallback.
Auto-delete of S3 objects optionally supported.
"""

import logging
import aiohttp
import os
import base64
from pylatexenc.latex2text import LatexNodes2Text
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeDefault
import asyncio
from aiogram.utils.keyboard import InlineKeyboardBuilder
from PIL import Image
from typing import Optional, Dict, Any
from uuid import uuid4
import boto3
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor

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
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "90"))     # JPEG quality (1-100)

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
    "Qwen3-coder": {"id": "qwen/qwen3-coder:free", "image_support": False},
    "Gemini-2.0-flash": {"id": "google/gemini-2.0-flash-exp:free", "image_support": True},
    "Qwen3": {"id": "qwen/qwen3-235b-a22b:free", "image_support": False},
    "GPT-5 mini": {"id": "openai/gpt-5-mini", "image_support": True},
}
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
user_models: Dict[int, str] = {}

# ---------- Utilities ----------
def convert_markdown_to_html(text: str) -> str:
    return LatexNodes2Text().latex_to_text(text)

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

def compress_image_high_quality(image_data: bytes, max_side: int = IMAGE_MAX_SIZE, quality: int = IMAGE_QUALITY) -> bytes:
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
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
        out = BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception:
        logging.exception("compress_image_high_quality error")
        return image_data

def estimate_tokens_from_bytes(n_bytes: int) -> int:
    """Rough heuristic: tokens ≈ bytes / 3"""
    return max(1, int(n_bytes / 3))

# ---------- S3 / Cloudflare R2 helpers ----------
_s3_client = None

def _make_s3_client_sync():
    """
    Create and return a boto3 S3 client (sync). Called inside thread executor.
    """
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

async def upload_bytes_to_s3_and_get_presigned_url(bytes_data: bytes, key: str, expires: int = S3_PRESIGNED_EXPIRATION) -> Optional[str]:
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
        BotCommand(command="/upload_test", description="Тест загрузки изображения (публичный URL)"),
    ]
    await bot.set_my_commands(cmds, scope=BotCommandScopeDefault())

@dp.message(Command("start"))
async def cmd_start(m: types.Message):
    await m.answer(
        "🤖 <b>AI Chat Bot</b>\n\n"
        "Отправьте текст или изображение. Для изображений бот загружает файл в S3 (Cloudflare R2) и отправляет presigned URL."
    )

@dp.message(Command("help"))
async def cmd_help(m: types.Message):
    help_text = (
        "🛠️ <b>Команды:</b>\n"
        "/start - Запустить бота\n"
        "/help - Помощь\n"
        "/models - Выбрать модель\n"
        "/currentmodel - Текущая модель\n"
        "/upload_test - Тест публичного URL\n\n"
        "<b>Доступные модели:</b>\n"
    )
    for model_name, model_data in AVAILABLE_MODELS.items():
        camera_icon = " 📷" if model_data["image_support"] else ""
        help_text += f"• {model_name}{camera_icon}\n"
    await m.answer(help_text)

@dp.message(Command("models"))
async def list_models(m: types.Message):
    await show_models_keyboard(m, m.from_user.id)

async def show_models_keyboard(message: types.Message, user_id: int, edit: bool = False):
    builder = InlineKeyboardBuilder()
    current_model_id = get_user_model(user_id)
    for model_name, model_data in AVAILABLE_MODELS.items():
        prefix = "✅ " if model_data["id"] == current_model_id else ""
        camera_icon = " 📷" if model_data["image_support"] else ""
        builder.button(text=f"{prefix}{model_name}{camera_icon}", callback_data=f"model_{model_data['id']}")
    builder.button(text="❌ Закрыть", callback_data="close_menu")
    builder.adjust(2, 2, 2)
    current_model_name = get_model_name(current_model_id)
    if edit:
        await message.edit_text(f"🛠️ <b>Выберите модель:</b>\nТекущая: {current_model_name}\n\n<i>Нажмите на модель</i>",
                                reply_markup=builder.as_markup())
    else:
        await message.answer(f"🛠️ <b>Выберите модель:</b>\nТекущая: {current_model_name}\n\n<i>Нажмите на модель</i>",
                             reply_markup=builder.as_markup())

@dp.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
    model_id = callback.data.split("_", 1)[1]
    user_id = callback.from_user.id
    model_exists = any(model_data["id"] == model_id for model_data in AVAILABLE_MODELS.values())
    if model_exists:
        user_models[user_id] = model_id
        await show_models_keyboard(callback.message, user_id, edit=True)
        model_name = get_model_name(model_id)
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
    model_name = get_model_name(model_id)
    image_support = "Да" if get_model_support_images(model_id) else "Нет"
    await m.answer(f"🔧 <b>Текущая модель:</b> {model_name}\n📸 <b>Поддержка изображений:</b> {image_support}")

@dp.message(Command("upload_test"))
async def cmd_upload_test(m: types.Message):
    """
    Test: sends a public image URL (Wikimedia) to the selected model to compare usage.
    """
    test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/1024px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    model_id = get_user_model(m.from_user.id)
    if not get_model_support_images(model_id):
        return await m.answer("Текущая модель не поддерживает изображения. Выберите модель с поддержкой изображений (/models).")
    status = await m.answer("<i>Тестовый запрос: отправляю публичный URL...</i>", parse_mode=ParseMode.HTML)
    content = [{"type":"text","text":"Опиши это изображение (тестовый публичный URL)"}, {"type":"image_url","image_url":{"url":test_url}}]
    payload = {"model": model_id, "messages":[{"role":"user","content":content}]}
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
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
    status_msg = await message.answer("<i>🤖 Обрабатываю изображение...</i>", parse_mode=ParseMode.HTML)
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(message.from_user.id)
        if not get_model_support_images(model_id):
            await status_msg.delete()
            return await message.answer(
                "⚠️ Текущая модель не поддерживает обработку изображений через URL.\n"
                "Отправка фото этой моделью невозможна. Пожалуйста, выберите модель с поддержкой изображений командой /models."
            )

        # compress to high-quality JPEG
        jpeg_bytes = compress_image_high_quality(image_bytes)
        compressed_n = len(jpeg_bytes)
        logging.info(f"Original bytes: {len(image_bytes)}; JPEG bytes: {compressed_n}; est tokens if inlined: {estimate_tokens_from_bytes(compressed_n)}")

        # prepare S3 key
        key = f"telegram_uploads/{uuid4().hex}.jpg"

        # upload to S3 and get presigned URL
        presigned_url = await upload_bytes_to_s3_and_get_presigned_url(jpeg_bytes, key, expires=S3_PRESIGNED_EXPIRATION)
        if not presigned_url:
            await status_msg.delete()
            logging.error("Failed to upload to S3 / generate presigned URL")
            return await message.answer("Не удалось загрузить изображение в облако (S3). Повторите позже или проверьте настройки.")

        logging.info(f"Uploaded to S3 key={key} presigned_url={presigned_url}")

        # schedule auto-delete if configured
        if S3_AUTO_DELETE_AFTER > 0:
            await schedule_s3_delete(key, delay=S3_AUTO_DELETE_AFTER)

        # Build payload with image_url only
        content = []
        content.append({"type":"text","text": caption or "Опиши это изображение"})
        content.append({"type":"image_url","image_url":{"url": presigned_url}})

        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"}
        payload = {"model": model_id, "messages":[{"role":"user","content":content}]}

        logging.info(f"Sending to OpenRouter model={model_id} payload=image_url (presigned S3)")
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    await status_msg.delete()
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status})")

        logging.info(f"OpenRouter usage: {data.get('usage')}")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        await status_msg.delete()
        # split if long
        model_name = get_model_name(model_id)
        response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted}"
        if len(response_text) > 4096:
            for i in range(0, len(response_text), 4096):
                await message.answer(response_text[i:i+4096])
        else:
            await message.answer(response_text)

    except Exception:
        logging.exception("process_image_message error")
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer("🚫 Ошибка при обработке изображения")

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
    status = await message.answer("<i>🤖 Обрабатываю...</i>", parse_mode=ParseMode.HTML)
    await bot.send_chat_action(message.chat.id, "typing")
    try:
        model_id = get_user_model(message.from_user.id)
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":"application/json"}
        payload = {"model": model_id, "messages":[{"role":"user","content": message.text}]}
        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    logging.error(f"OpenRouter returned non-json: status={resp.status} text={text[:400]}")
                    await status.delete()
                    return await message.answer(f"⚠️ Ошибка OpenRouter ({resp.status})")
        logging.info(f"OpenRouter usage: {data.get('usage')}")
        ai_response = data['choices'][0]['message']['content']
        formatted = convert_markdown_to_html(ai_response)
        await status.delete()
        # split if long
        response = f"🧠 <b>{get_model_name(model_id)}</b>:\n\n{formatted}"
        if len(response) > 4096:
            for i in range(0, len(response), 4096):
                await message.answer(response[i:i+4096])
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
