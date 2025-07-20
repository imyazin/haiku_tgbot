import os
import logging
import asyncio
import threading
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from training import HaikuGenerator
from dataset import get_haiku_dataset

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class HaikuTelegramBot:
    def __init__(self, token):
        self.token = token
        self.generator = HaikuGenerator()
        self.model_loaded = False
        self.training_in_progress = False
        self.application = None
        
        # Статистика бота
        self.stats = {
            'total_requests': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'users': set()
        }
    
    async def load_or_train_model(self):
        """Загрузка существующей модели или обучение новой"""
        try:
            # Попытка загрузить существующую модель
            if os.path.exists("haiku_model.h5") and os.path.exists("tokenizer.pkl"):
                logger.info("Загрузка существующей модели...")
                self.generator.load_model()
                self.model_loaded = True
                logger.info("Модель успешно загружена!")
            else:
                logger.info("Модель не найдена. Начинаем обучение...")
                await self.train_model()
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            await self.train_model()
    
    async def train_model(self):
        """Обучение модели"""
        self.training_in_progress = True
        try:
            logger.info("Начинаем обучение модели на датасете...")
            
            # Загрузка датасета
            haiku_data = get_haiku_dataset()
            logger.info(f"Загружено {len(haiku_data)} хокку для обучения")
            
            # Обучение в отдельном потоке
            def train_in_thread():
                self.generator.train(haiku_data, epochs=150, batch_size=16)
                self.generator.save_model()
                self.model_loaded = True
                self.training_in_progress = False
                logger.info("Обучение завершено и модель сохранена!")
            
            # Запуск обучения в отдельном потоке
            training_thread = threading.Thread(target=train_in_thread)
            training_thread.start()
            
            # Ждем завершения обучения (с таймаутом)
            max_wait_time = 3600  # 1 час максимум
            wait_time = 0
            while self.training_in_progress and wait_time < max_wait_time:
                await asyncio.sleep(10)
                wait_time += 10
                if wait_time % 60 == 0:  # Каждую минуту
                    logger.info(f"Обучение продолжается... ({wait_time//60} мин)")
            
        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {e}")
            self.training_in_progress = False
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /start"""
        user_id = update.effective_user.id
        self.stats['users'].add(user_id)
        
        welcome_text = """
🌸 Добро пожаловать в бота для генерации хокку! 🌸

Я умею создавать красивые японские стихотворения по вашей первой строке.

📝 Как использовать:
• Напишите первую строку хокку
• Я придумаю продолжение в традиционном стиле

🎯 Команды:
/start - показать это сообщение
/help - подробная помощь
/example - примеры хокку
/stats - статистика бота
/retrain - переобучить модель (только для админа)

💡 Совет: первая строка должна содержать 5 слогов для лучшего результата!
        """
        
        keyboard = [
            [InlineKeyboardButton("📖 Примеры", callback_data="examples")],
            [InlineKeyboardButton("❓ Помощь", callback_data="help")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка команды /help"""
        help_text = """
📚 ПОДРОБНАЯ ПОМОЩЬ

🎋 Что такое хокку?
Хокку - традиционная форма японской поэзии из 3 строк:
• 1-я строка: 5 слогов
• 2-я строка: 7 слогов  
• 3-я строка: 5 слогов

🖋️ Как пользоваться ботом:
1. Напишите первую строку (желательно 5 слогов)
2. Бот сгенерирует продолжение
3. Наслаждайтесь получившимся хокку!

✨ Примеры хороших первых строк:
• "Утренний туман" (5 слогов)
• "Весенний дождик льет" (6 слогов)
• "Осенние листья" (6 слогов)

🎨 Темы для вдохновения:
• Природа и времена года
• Животные и растения
• Чувства и настроения
• Городская жизнь
• Философские размышления

⚙️ Технические особенности:
• Модель обучена на 150+ русских хокку
• Использует нейросеть LSTM
• Генерация занимает 1-3 секунды
        """
        
        # Проверяем, откуда пришел вызов
        if update.callback_query:
            await update.callback_query.edit_message_text(help_text)
        else:
            await update.message.reply_text(help_text)
    
    async def example_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать примеры хокку"""
        examples = [
            ("Утренний туман", "Стелется над рекой\nПризрачным покровом"),
            ("Весенний дождик льет", "По листьям тихо бьет\nЗемля благоухает"),
            ("Осенние листья", "Кружатся в вихре ветра\nЗолотой ковер"),
            ("Первый снег упал", "На голые ветви\nМир стал белоснежным"),
            ("Летний зной палит", "Цикады песни поют\nВ тени прохлада")
        ]
        
        example_text = "📖 ПРИМЕРЫ ХОККУ:\n\n"
        for i, (first_line, continuation) in enumerate(examples, 1):
            example_text += f"{i}. {first_line}\n{continuation}\n\n"
        
        example_text += "💡 Попробуйте написать свою первую строку!"
        
        keyboard = [
            [InlineKeyboardButton("🎲 Случайная строка", callback_data="random_start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Проверяем, откуда пришел вызов
        if update.callback_query:
            await update.callback_query.edit_message_text(example_text, reply_markup=reply_markup)
        else:
            await update.message.reply_text(example_text, reply_markup=reply_markup)
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показать статистику бота"""
        stats_text = f"""
📊 СТАТИСТИКА БОТА

👥 Пользователей: {len(self.stats['users'])}
📝 Всего запросов: {self.stats['total_requests']}
✅ Успешных генераций: {self.stats['successful_generations']}
❌ Ошибок: {self.stats['failed_generations']}
🎯 Успешность: {(self.stats['successful_generations']/max(1,self.stats['total_requests'])*100):.1f}%

🤖 Статус модели: {'✅ Загружена' if self.model_loaded else '❌ Не загружена'}
🏋️ Обучение: {'🔄 В процессе' if self.training_in_progress else '✅ Завершено'}

⏰ Время работы: {datetime.now().strftime('%H:%M:%S')}
        """
        
        # Проверяем, откуда пришел вызов
        if update.callback_query:
            await update.callback_query.edit_message_text(stats_text)
        else:
            await update.message.reply_text(stats_text)
    
    async def generate_haiku(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Генерация хокку по первой строке"""
        if not self.model_loaded:
            if self.training_in_progress:
                await update.message.reply_text(
                    "🤖 Модель еще обучается, пожалуйста подождите...\n"
                    "⏱️ Обучение может занять несколько минут."
                )
            else:
                await update.message.reply_text(
                    "❌ Модель не загружена. Попробуйте позже или обратитесь к администратору."
                )
            return
        
        first_line = update.message.text.strip()
        user_id = update.effective_user.id
        
        # Обновляем статистику
        self.stats['users'].add(user_id)
        self.stats['total_requests'] += 1
        
        # Проверяем длину строки
        if len(first_line) > 100:
            await update.message.reply_text(
                "❌ Первая строка слишком длинная. Попробуйте что-то покороче."
            )
            return
        
        if len(first_line) < 3:
            await update.message.reply_text(
                "❌ Первая строка слишком короткая. Напишите хотя бы несколько слов."
            )
            return
        
        # Показываем, что бот печатает
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Генерируем хокку
            completion = self.generator.generate_haiku_completion(
                first_line, 
                max_length=15, 
                temperature=0.8
            )
            
            if completion and completion.strip():
                # Форматируем результат
                haiku_text = f"🌸 **Ваше хокку:**\n\n"
                haiku_text += f"*{first_line}*\n"
                haiku_text += f"{completion}\n\n"
                haiku_text += "✨ *Создано нейросетью*"
                
                # Кнопки для взаимодействия
                keyboard = [
                    [InlineKeyboardButton("🔄 Другой вариант", callback_data=f"regenerate:{first_line}")],
                    [InlineKeyboardButton("💾 Сохранить", callback_data=f"save:{first_line}:{completion}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    haiku_text, 
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
                self.stats['successful_generations'] += 1
            else:
                await update.message.reply_text(
                    "😅 Не получилось создать хокку. Попробуйте другую первую строку."
                )
                self.stats['failed_generations'] += 1
                
        except Exception as e:
            logger.error(f"Ошибка при генерации хокку: {e}")
            await update.message.reply_text(
                "😵 Произошла ошибка при создании хокку. Попробуйте еще раз."
            )
            self.stats['failed_generations'] += 1
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатий на кнопки"""
        query = update.callback_query
        await query.answer()
        
        if query.data == "examples":
            await self.example_command(update, context)
        elif query.data == "help":
            await self.help_command(update, context)
        elif query.data == "stats":
            await self.stats_command(update, context)
        elif query.data == "random_start":
            random_starts = [
                "Утренняя роса", "Вечерний закат", "Зимний ветер",
                "Летний дождь", "Осенний лист", "Весенний цветок",
                "Горная тропа", "Морские волны", "Лесная тишь",
                "Городской шум", "Детский смех", "Старый дом"
            ]
            import random
            random_start = random.choice(random_starts)
            await query.edit_message_text(
                f"🎲 Случайная первая строка: **{random_start}**\n\n"
                f"Отправьте эту строку для генерации хокку!",
                parse_mode='Markdown'
            )
        elif query.data.startswith("regenerate:"):
            first_line = query.data.split(":", 1)[1]
            
            # Показываем, что бот печатает
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            try:
                # Генерируем новое хокку
                completion = self.generator.generate_haiku_completion(
                    first_line, 
                    max_length=15, 
                    temperature=0.9  # Увеличиваем температуру для большего разнообразия
                )
                
                if completion and completion.strip():
                    # Форматируем результат
                    haiku_text = f"🌸 **Новый вариант хокку:**\n\n"
                    haiku_text += f"*{first_line}*\n"
                    haiku_text += f"{completion}\n\n"
                    haiku_text += "✨ *Создано нейросетью*"
                    
                    # Кнопки для взаимодействия
                    keyboard = [
                        [InlineKeyboardButton("🔄 Еще вариант", callback_data=f"regenerate:{first_line}")],
                        [InlineKeyboardButton("💾 Сохранить", callback_data=f"save:{first_line}:{completion}")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.message.reply_text(
                        haiku_text, 
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    
                    self.stats['successful_generations'] += 1
                else:
                    await query.message.reply_text(
                        "😅 Не получилось создать новый вариант. Попробуйте еще раз."
                    )
                    self.stats['failed_generations'] += 1
                    
            except Exception as e:
                logger.error(f"Ошибка при регенерации хокку: {e}")
                await query.message.reply_text(
                    "😵 Произошла ошибка при создании нового варианта. Попробуйте еще раз."
                )
                self.stats['failed_generations'] += 1
                
        elif query.data.startswith("save:"):
            parts = query.data.split(":", 2)
            if len(parts) == 3:
                first_line, completion = parts[1], parts[2]
                saved_haiku = f"{first_line}\n{completion}"
                await query.edit_message_text(
                    f"💾 **Сохраненное хокку:**\n\n{saved_haiku}\n\n"
                    f"📋 Готово для копирования!",
                    parse_mode='Markdown'
                )
    
    async def retrain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Переобучение модели (только для админов)"""
        # Здесь можно добавить проверку на админа
        admin_ids = [123456789]  # Замените на реальные ID администраторов
        
        if update.effective_user.id not in admin_ids:
            await update.message.reply_text("❌ У вас нет прав для выполнения этой команды.")
            return
        
        if self.training_in_progress:
            await update.message.reply_text("🔄 Обучение уже в процессе...")
            return
        
        await update.message.reply_text("🚀 Начинаем переобучение модели...")
        await self.train_model()
        await update.message.reply_text("✅ Модель переобучена!")
    
    def run(self):
        """Запуск бота"""
        # Создаем приложение
        self.application = Application.builder().token(self.token).build()
        
        # Регистрируем обработчики
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("example", self.example_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("retrain", self.retrain_command))
        self.application.add_handler(CallbackQueryHandler(self.button_handler))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.generate_haiku))
        
        # Загружаем модель при запуске
        async def post_init(application):
            await self.load_or_train_model()
        
        self.application.post_init = post_init
        
        # Запускаем бота
        print("🚀 Запускаем Telegram бота для генерации хокку...")
        self.application.run_polling()

# Основная функция
def main():
    # Получаем токен бота из переменной окружения
    TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    
    if not TOKEN:
        print("❌ Ошибка: не установлена переменная окружения TELEGRAM_BOT_TOKEN")
        print("Получите токен у @BotFather и установите переменную:")
        print("export TELEGRAM_BOT_TOKEN='your_token_here'")
        return
    
    # Создаем и запускаем бота
    bot = HaikuTelegramBot(TOKEN)
    bot.run()

if __name__ == "__main__":
    main()