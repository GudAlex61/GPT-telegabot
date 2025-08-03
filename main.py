import logging
import aiohttp
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeDefault
import asyncio
from aiogram.utils.keyboard import InlineKeyboardBuilder

# Проверка и загрузка .env
env_path = Path('.') / '.env'
if not env_path.exists():
    print(f"❌ Файл .env не найден по пути: {env_path.absolute()}")
    print("Создайте файл .env с содержимым:")
    print("TELEGRAM_BOT_TOKEN=ваш_токен_бота")
    print("OPENROUTER_API_KEY=sk-or-ваш_ключ")
    exit(1)

load_dotenv(env_path)

API_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY3')

# Настройка бота
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Доступные модели (бесплатные)
AVAILABLE_MODELS = {
    "Deepseek-V3": "deepseek/deepseek-chat-v3-0324:free",
    "Deepseek-R1": "deepseek/deepseek-r1-0528:free",
    "Qwen3-coder": "qwen/qwen3-coder:free",
    "Gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
    "Qwen3": "qwen/qwen3-235b-a22b:free",
    "GPT-4o mini": "openai/gpt-4o-mini"
}

# Модель по умолчанию
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Хранилище выбранных моделей (в памяти)
user_models = {}


def get_user_model(user_id):
    """Получить модель для пользователя или модель по умолчанию"""
    return user_models.get(user_id, DEFAULT_MODEL)


async def set_main_menu():
    """Создаем меню команд"""
    main_menu_commands = [
        BotCommand(command='/start', description='Запустить бота'),
        BotCommand(command='/help', description='Помощь и команды'),
        BotCommand(command='/models', description='Выбрать модель ИИ'),
        BotCommand(command='/currentmodel', description='Текущая модель'),
    ]
    await bot.set_my_commands(main_menu_commands, scope=BotCommandScopeDefault())


@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "🤖 <b>Добро пожаловать в AI Chat Bot!</b>\n\n"
        "Я использую различные языковые модели для ответов на ваши вопросы.\n\n"
        "🔧 <b>Основные команды:</b>\n"
        "/models - Выбрать модель ИИ\n"
        "/currentmodel - Текущая модель\n"
        "/help - Справка по командам\n\n"
        "Просто отправьте сообщение, чтобы начать диалог!"
    )


@dp.message(Command("help"))
async def help_command(message: types.Message):
    """Справка по командам"""
    help_text = (
        "🛠️ <b>Список доступных команд:</b>\n\n"
        "/start - Запустить бота\n"
        "/help - Показать это сообщение\n"
        "/models - Выбрать модель ИИ из списка\n"
        "/currentmodel - Показать текущую модель\n\n"
        "📚 <b>Доступные модели:</b>\n"
    )

    # Добавляем список моделей
    for model_name in AVAILABLE_MODELS.keys():
        help_text += f"• {model_name}\n"

    help_text += "\nПросто отправьте текст, чтобы задать вопрос выбранной модели ИИ."

    await message.answer(help_text)


@dp.message(Command("models"))
async def list_models(message: types.Message):
    """Показать список доступных моделей с галочкой у текущей"""
    await show_models_keyboard(message, message.from_user.id)


async def show_models_keyboard(message: types.Message, user_id: int, edit: bool = False):
    """Показать/обновить клавиатуру с моделями"""
    builder = InlineKeyboardBuilder()

    current_model_id = get_user_model(user_id)

    for model_name, model_id in AVAILABLE_MODELS.items():
        # Добавляем галочку к текущей выбранной модели
        prefix = "✅ " if model_id == current_model_id else ""
        builder.button(text=f"{prefix}{model_name}", callback_data=f"model_{model_id}")

    # Добавляем кнопку закрытия
    builder.button(text="❌ Закрыть", callback_data="close_menu")

    builder.adjust(2, 2, 1)  # Распределение кнопок: 2 в первом ряду, 2 во втором, 1 в третьем

    current_model_name = next((k for k, v in AVAILABLE_MODELS.items() if v == current_model_id), current_model_id)

    if edit:
        # Редактируем существующее сообщение
        await message.edit_text(
            f"🛠️ <b>Выберите модель:</b>\n"
            f"🔧 Текущая: {current_model_name}\n\n"
            f"<i>Нажмите на модель для выбора</i>",
            reply_markup=builder.as_markup()
        )
    else:
        # Отправляем новое сообщение
        await message.answer(
            f"🛠️ <b>Выберите модель:</b>\n"
            f"🔧 Текущая: {current_model_name}\n\n"
            f"<i>Нажмите на модель для выбора</i>",
            reply_markup=builder.as_markup()
        )


@dp.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
    """Обработка выбора модели"""
    model_id = callback.data.split("_", 1)[1]
    user_id = callback.from_user.id

    if model_id in AVAILABLE_MODELS.values():
        user_models[user_id] = model_id
        # Обновляем сообщение с новым выбором
        await show_models_keyboard(callback.message, user_id, edit=True)
        await callback.answer(f"Модель изменена")
    else:
        await callback.answer("⚠️ Неизвестная модель", show_alert=True)


@dp.callback_query(F.data == "close_menu")
async def close_menu_callback(callback: types.CallbackQuery):
    """Закрытие меню выбора модели"""
    await callback.message.delete()
    await callback.answer("Меню закрыто")

@dp.message(Command("currentmodel"))
async def current_model(message: types.Message):
    """Показать текущую модель"""
    model_id = get_user_model(message.from_user.id)
    model_name = next((k for k, v in AVAILABLE_MODELS.items() if v == model_id), model_id)
    await message.answer(f"🔧 Текущая модель: <b>{model_name}</b>")


@dp.message()
async def handle_message(message: types.Message):
    if not message.text:
        return await message.answer("Отправьте текстовый вопрос")

    # Отправляем статус "Обработка запроса..."
    status_msg = await message.answer(
        f"<i>🤖 Обрабатываю ваш запрос... Пожалуйста, подождите немного.</i>",
        parse_mode=ParseMode.HTML
    )

    # Показываем статус "печатает" в чате
    await bot.send_chat_action(message.chat.id, "typing")

    try:
        # Получаем модель для текущего пользователя
        model_id = get_user_model(message.from_user.id)

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://your-telegram-bot.com",
            "X-Title": "Telegram Bot",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": message.text}]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    # Удаляем статусное сообщение перед отправкой ошибки
                    await status_msg.delete()
                    return await message.answer(f"⚠️ Ошибка API: {error[:200]}...")

                data = await response.json()

                # Удаляем статусное сообщение
                await status_msg.delete()

                # Форматируем ответ с указанием модели
                model_name = next((k for k, v in AVAILABLE_MODELS.items() if v == model_id), model_id)
                response_text = f"🧠 <b>{model_name}</b>:\n\n"
                response_text += data['choices'][0]['message']['content']

                await message.answer(response_text)

    except Exception as e:
        # Удаляем статусное сообщение в случае ошибки
        await status_msg.delete()
        await message.answer(f"🚫 Ошибка: {str(e)}")


async def main():
    # Устанавливаем меню команд
    await set_main_menu()

    print("✅ Бот запущен!")
    print(f"Токен бота: {'установлен' if API_TOKEN else 'НЕТ'}")
    print(f"Ключ OpenRouter: {'установлен' if OPENROUTER_API_KEY else 'НЕТ'}")
    print(f"Доступные модели: {', '.join(AVAILABLE_MODELS.keys())}")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())