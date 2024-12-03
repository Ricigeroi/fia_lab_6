from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

# Команда /start
async def start(update: Update, context):
    await update.message.reply_text("Привет! Я простой бот. Напиши мне что-нибудь!")

# Ответ на любое сообщение
async def echo(update: Update, context):
    user_message = update.message.text
    await update.message.reply_text(f"Ты написал: {user_message}")

# Основной код
if __name__ == "__main__":
    # Вставьте ваш токен сюда
    TOKEN = "7833443122:AAGdtX0h6BYLF765IVBBwGPiWsWDrVpYCGY"

    # Создаем приложение
    app = ApplicationBuilder().token(TOKEN).build()

    # Добавляем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Запускаем бота
    print("Бот запущен...")
    app.run_polling()
