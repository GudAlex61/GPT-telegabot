import logging
import aiohttp
import os
import json
import re
import html
import base64
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode, ContentType
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import BotCommand, BotCommandScopeDefault, URLInputFile
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

# Проверка наличия ключей
if not API_TOKEN or not OPENROUTER_API_KEY:
    print("❌ Ошибка: Не найдены необходимые переменные окружения")
    print("Убедитесь что в .env есть TELEGRAM_BOT_TOKEN и OPENROUTER_API_KEY")
    exit(1)

# Настройка бота
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# Доступные модели (мультимодальные помечены как image_support=True)
AVAILABLE_MODELS = {
    "Deepseek-V3": {
        "id": "deepseek/deepseek-chat-v3-0324:free",
        "image_support": False
    },
    "Deepseek-R1": {
        "id": "deepseek/deepseek-r1-0528:free",
        "image_support": False
    },
    "Qwen3-coder": {
        "id": "qwen/qwen3-coder:free",
        "image_support": False
    },
    "Gemini-2.0-flash": {
        "id": "google/gemini-2.0-flash-exp:free",
        "image_support": True
    },
    "Qwen3": {
        "id": "qwen/qwen3-235b-a22b:free",
        "image_support": False
    },
    "GPT-4o mini": {
        "id": "openai/gpt-4o-mini",
        "image_support": True
    }
}

# Модель по умолчанию
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Хранилище выбранных моделей (в памяти)
user_models = {}


def convert_markdown_to_html(text: str) -> str:
    """Конвертирует Markdown-разметку в HTML-теги"""
    # Экранируем HTML-символы
    text = html.escape(text)
    
    # Обрабатываем блоки кода с указанием языка
    text = re.sub(
        r'```(\w+)?\s*\n(.+?)```', 
        r'<pre><code>\2</code></pre>', 
        text, 
        flags=re.DOTALL
    )
    
    # Обрабатываем блоки кода без указания языка
    text = re.sub(
        r'```([^`]+?)```', 
        r'<pre><code>\1</code></pre>', 
        text, 
        flags=re.DOTALL
    )
    
    # Обрабатываем инлайн-код
    text = re.sub(
        r'`([^`]+?)`', 
        r'<code>\1</code>', 
        text
    )
    
    # Жирный текст
    text = re.sub(
        r'\*\*([^*]+?)\*\*', 
        r'<b>\1</b>', 
        text
    )
    
    # Курсив
    text = re.sub(
        r'\*([^*]+?)\*', 
        r'<i>\1</i>', 
        text
    )
    
    # Обработка заголовков H1-H6
    # H1: # Заголовок
    text = re.sub(
        r'^# (.+)$', 
        r'<b><u>\1</u></b>', 
        text, 
        flags=re.MULTILINE
    )
    
    # H3: ### Заголовок
    text = re.sub(
        r'^### (.+)$', 
        r'<b>\1</b>', 
        text, 
        flags=re.MULTILINE
    )

        # Специальные математические символы (добавляем пробелы для лучшей читаемости)
    math_symbols = {
        r'\\int': '∫',
        r'\\sum': '∑',
        r'\\prod': '∏',
        r'\\pm': '±',
        r'\\mp': '∓',
        r'\\infty': '∞',
        r'\\cdot': '·',
        r'\\times': '×',
        r'\\div': '÷',
        r'\\sqrt': '√',
        r'\\pi': 'π',
        r'\\alpha': 'α',
        r'\\beta': 'β',
        r'\\gamma': 'γ',
        r'\\delta': 'δ',
        r'\\epsilon': 'ε',
        r'\\zeta': 'ζ',
        r'\\eta': 'η',
        r'\\theta': 'θ',
        r'\\lambda': 'λ',
        r'\\mu': 'μ',
        r'\\xi': 'ξ',
        r'\\rho': 'ρ',
        r'\\sigma': 'σ',
        r'\\tau': 'τ',
        r'\\phi': 'φ',
        r'\\psi': 'ψ',
        r'\\omega': 'ω',
        r'\\Delta': 'Δ',
        r'\\Gamma': 'Γ',
        r'\\Theta': 'Θ',
        r'\\Lambda': 'Λ',
        r'\\Sigma': 'Σ',
        r'\\Phi': 'Φ',
        r'\\Psi': 'Ψ',
        r'\\Omega': 'Ω',
        r'\\partial': '∂',
        r'\\nabla': '∇',
        r'\\forall': '∀',
        r'\\exists': '∃',
        r'\\nexists': '∄',
        r'\\emptyset': '∅',
        r'\\in': '∈',
        r'\\notin': '∉',
        r'\\subset': '⊂',
        r'\\supset': '⊃',
        r'\\subseteq': '⊆',
        r'\\supseteq': '⊇',
        r'\\cap': '∩',
        r'\\cup': '∪',
        r'\\land': '∧',
        r'\\lor': '∨',
        r'\\neg': '¬',
        r'\\equiv': '≡',
        r'\\approx': '≈',
        r'\\propto': '∝',
        r'\\perp': '⊥',
        r'\\angle': '∠',
        r'\\therefore': '∴',
        r'\\because': '∵',
    }
    
    for pattern, replacement in math_symbols.items():
        text = re.sub(pattern, replacement, text)
    
    # Дроби вида \frac{a}{b} → a/b
    text = re.sub(
        r'\\frac\s*{([^}]+)}\s*{([^}]+)}', 
        r'<i>\1</i>/<i>\2</i>', 
        text
    )
    
    # Интегралы с пределами
    text = re.sub(
        r'\\int\s*_{([^}]+)}\s*^{([^}]+)}', 
        r'∫<sub>\1</sub><sup>\2</sup>', 
        text
    )
    
    # Суммы и произведения с пределами
    text = re.sub(
        r'\\sum\s*_{([^}]+)}\s*^{([^}]+)}', 
        r'∑<sub>\1</sub><sup>\2</sup>', 
        text
    )
    
    text = re.sub(
        r'\\prod\s*_{([^}]+)}\s*^{([^}]+)}', 
        r'∏<sub>\1</sub><sup>\2</sup>', 
        text
    )
    
    # Индексы и степени
    text = re.sub(
        r'\^(\w+)', 
        r'<sup>\1</sup>', 
        text
    )
    
    text = re.sub(
        r'_(\w+)', 
        r'<sub>\1</sub>', 
        text
    )
    
    # Греческие буквы и другие символы в тексте
    text = re.sub(
        r'\\text\s*{([^}]+)}', 
        r'\1', 
        text
    )
    
    # Добавляем пробелы вокруг операторов для лучшей читаемости
    operators = [r'\+', r'-', r'=', r'<', r'>', r'\\leq', r'\\geq', r'\\neq']
    for op in operators:
        text = re.sub(f'({op})', r' \1 ', text)
    
    return text

def get_user_model(user_id):
    """Получить модель для пользователя или модель по умолчанию"""
    return user_models.get(user_id, DEFAULT_MODEL)

def get_model_name(model_id):
    """Получить название модели по её ID"""
    for name, data in AVAILABLE_MODELS.items():
        if data["id"] == model_id:
            return name
    return model_id

def get_model_support_images(model_id):
    """Проверить, поддерживает ли модель изображения"""
    for data in AVAILABLE_MODELS.values():
        if data["id"] == model_id:
            return data["image_support"]
    return False


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
        "📸 <b>Поддержка изображений:</b>\n"
        "Вы можете отправлять изображения! Некоторые модели (GPT-4o mini, Gemini) поддерживают анализ изображений.\n\n"
        "Просто отправьте сообщение или изображение, чтобы начать диалог!"
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

    # Добавляем список моделей с иконкой камеры для поддерживающих изображения
    for model_name, model_data in AVAILABLE_MODELS.items():
        camera_icon = " 📷" if model_data["image_support"] else ""
        help_text += f"• {model_name}{camera_icon}\n"

    help_text += (
        "\n📸 <b>Поддержка изображений:</b>\n"
        "Модели с иконкой камеры могут анализировать изображения.\n\n"
        "Просто отправьте текст или изображение, чтобы задать вопрос выбранной модели ИИ."
    )

    await message.answer(help_text)


@dp.message(Command("models"))
async def list_models(message: types.Message):
    """Показать список доступных моделей с галочкой у текущей"""
    await show_models_keyboard(message, message.from_user.id)


async def show_models_keyboard(message: types.Message, user_id: int, edit: bool = False):
    """Показать/обновить клавиатуру с моделями"""
    builder = InlineKeyboardBuilder()

    current_model_id = get_user_model(user_id)

    for model_name, model_data in AVAILABLE_MODELS.items():
        # Добавляем галочку к текущей выбранной модели
        prefix = "✅ " if model_data["id"] == current_model_id else ""
        # Добавляем иконку камеры для моделей с поддержкой изображений
        camera_icon = " 📷" if model_data["image_support"] else ""
        builder.button(text=f"{prefix}{model_name}{camera_icon}", callback_data=f"model_{model_data['id']}")

    # Добавляем кнопку закрытия
    builder.button(text="❌ Закрыть", callback_data="close_menu")

    builder.adjust(2, 2, 2)  # Распределение кнопок: 2 в первом ряду, 2 во втором, 2 в третьем

    current_model_name = get_model_name(current_model_id)

    if edit:
        # Редактируем существующее сообщение
        await message.edit_text(
            f"🛠️ <b>Выберите модель:</b>\n"
            f"🔧 Текущая: {current_model_name}\n"
            f"📸 Модели с иконкой камеры поддерживают изображения\n\n"
            f"<i>Нажмите на модель для выбора</i>",
            reply_markup=builder.as_markup()
        )
    else:
        # Отправляем новое сообщение
        await message.answer(
            f"🛠️ <b>Выберите модель:</b>\n"
            f"🔧 Текущая: {current_model_name}\n"
            f"📸 Модели с иконкой камеры поддерживают изображения\n\n"
            f"<i>Нажмите на модель для выбора</i>",
            reply_markup=builder.as_markup()
        )


@dp.callback_query(F.data.startswith("model_"))
async def set_model_callback(callback: types.CallbackQuery):
    """Обработка выбора модели"""
    model_id = callback.data.split("_", 1)[1]
    user_id = callback.from_user.id

    # Проверяем, существует ли такая модель
    model_exists = any(model_data["id"] == model_id for model_data in AVAILABLE_MODELS.values())
    
    if model_exists:
        user_models[user_id] = model_id
        # Обновляем сообщение с новым выбором
        await show_models_keyboard(callback.message, user_id, edit=True)
        model_name = get_model_name(model_id)
        await callback.answer(f"Модель изменена на {model_name}")
    else:
        await callback.answer("⚠️ Неизвестная модель", show_alert=True)


@dp.callback_query(F.data == "close_menu")
async def close_menu_callback(callback: types.CallbackQuery):
    """Закрытие меню выбора модели"""
    try:
        await callback.message.delete()
    except Exception:
        pass  # Игнорируем ошибку, если сообщение уже удалено
    await callback.answer("Меню закрыто")

@dp.message(Command("currentmodel"))
async def current_model(message: types.Message):
    """Показать текущую модель"""
    model_id = get_user_model(message.from_user.id)
    model_name = get_model_name(model_id)
    image_support = "Да" if get_model_support_images(model_id) else "Нет"
    await message.answer(
        f"🔧 <b>Текущая модель:</b> {model_name}\n"
        f"📸 <b>Поддержка изображений:</b> {image_support}"
    )


async def process_image_message(message: types.Message, image_data: bytes, caption: str = ""):
    """Обработка сообщения с изображением"""
    # Отправляем статус "Обработка запроса..."
    status_msg = await message.answer(
        f"<i>🤖 Обрабатываю ваше изображение... Пожалуйста, подождите.</i>",
        parse_mode=ParseMode.HTML
    )

    # Показываем статус "печатает" в чате
    await bot.send_chat_action(message.chat.id, "typing")

    try:
        # Получаем модель для текущего пользователя
        model_id = get_user_model(message.from_user.id)
        
        # Проверяем, поддерживает ли модель изображения
        if not get_model_support_images(model_id):
            await status_msg.delete()
            return await message.answer(
                "⚠️ <b>Текущая модель не поддерживает анализ изображений</b>\n\n"
                "Используйте команду /models и выберите модель с иконкой камеры 📷"
            )

        # Конвертируем изображение в base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        mime_type = "image/jpeg"  # Telegram обычно использует JPEG

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://your-telegram-bot.com",
            "X-Title": "Telegram Bot",
            "Content-Type": "application/json"
        }

        # Формируем контент для мультимодального запроса
        content = []
        
        # Добавляем текст, если есть подпись
        if caption:
            content.append({"type": "text", "text": caption})
        else:
            content.append({"type": "text", "text": "Опиши это изображение"})
        
        # Добавляем изображение
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": content}]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    # Пытаемся удалить статусное сообщение
                    try:
                        await status_msg.delete()
                    except Exception:
                        pass
                    return await message.answer(f"⚠️ Ошибка API ({response.status}): {error[:200]}...")

                data = await response.json()
                ai_response = data['choices'][0]['message']['content']

                # Пытаемся удалить статусное сообщение
                try:
                    await status_msg.delete()
                except Exception:
                    pass

                # Конвертируем Markdown в HTML
                formatted_response = convert_markdown_to_html(ai_response)

                # Форматируем ответ с указанием модели
                model_name = get_model_name(model_id)
                response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted_response}"

                # Разбиваем длинные сообщения на части
                if len(response_text) > 4096:
                    for i in range(0, len(response_text), 4096):
                        part = response_text[i:i+4096]
                        await message.answer(part)
                else:
                    await message.answer(response_text)

    except Exception as e:
        # Пытаемся удалить статусное сообщение
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer(f"🚫 Ошибка: {str(e)}")


@dp.message(F.content_type == ContentType.PHOTO)
async def handle_photo(message: types.Message):
    """Обработка фото"""
    # Получаем фото с самым высоким разрешением
    photo = message.photo[-1]
    
    # Скачиваем фото
    file = await bot.get_file(photo.file_id)
    image_data = await bot.download_file(file.file_path)
    
    # Обрабатываем изображение
    await process_image_message(message, image_data.read(), message.caption)


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
                    # Пытаемся удалить статусное сообщение
                    try:
                        await status_msg.delete()
                    except Exception:
                        pass
                    return await message.answer(f"⚠️ Ошибка API ({response.status}): {error[:200]}...")

                data = await response.json()
                ai_response = data['choices'][0]['message']['content']

                # Пытаемся удалить статусное сообщение
                try:
                    await status_msg.delete()
                except Exception:
                    pass

                # Конвертируем Markdown в HTML
                formatted_response = convert_markdown_to_html(ai_response)

                # Форматируем ответ с указанием модели
                model_name = get_model_name(model_id)
                response_text = f"🧠 <b>{model_name}</b>:\n\n{formatted_response}"

                # Разбиваем длинные сообщения на части
                if len(response_text) > 4096:
                    for i in range(0, len(response_text), 4096):
                        part = response_text[i:i+4096]
                        await message.answer(part)
                else:
                    await message.answer(response_text)

    except Exception as e:
        # Пытаемся удалить статусное сообщение
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.answer(f"🚫 Ошибка: {str(e)}")


async def main():
    # Устанавливаем меню команд
    await set_main_menu()

    print("✅ Бот запущен!")
    print(f"Токен бота: {'установлен' if API_TOKEN else 'НЕТ'}")
    print(f"Ключ OpenRouter: {'установлен' if OPENROUTER_API_KEY else 'НЕТ'}")
    print(f"Доступные модели: {', '.join(AVAILABLE_MODELS.keys())}")
    print(f"Модели с поддержкой изображений:")
    for name, data in AVAILABLE_MODELS.items():
        if data["image_support"]:
            print(f"  - {name}")
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())