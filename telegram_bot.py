# telegram_bot.py

import os
import threading
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)

# حالة البوت
BOT_MODE = "DEMO"     # أو LIVE
BOT_RUNNING = False

# كولباكات يتم حقنها من main.py
RUN_CALLBACK = None
STOP_CALLBACK = None

# قراءة التوكن من متغير بيئة
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 مرحبًا i_x!\n"
        "هذا بوت التحكم في نظام التداول.\n\n"
        f"الوضع الحالي: {BOT_MODE}\n"
        "الأوامر:\n"
        "/mode demo – وضع تجريبي\n"
        "/mode live – وضع حقيقي\n"
        "/run – تشغيل البوت\n"
        "/stop – إيقاف البوت\n"
        "/status – حالة البوت"
    )


async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_MODE

    if not context.args:
        await update.message.reply_text(f"الوضع الحالي: {BOT_MODE}")
        return

    m = context.args[0].lower()

    if m == "demo":
        BOT_MODE = "DEMO"
    elif m == "live":
        BOT_MODE = "LIVE"
    else:
        await update.message.reply_text("❌ الوضع غير معروف. استخدم demo أو live")
        return

    await update.message.reply_text(f"✅ تم تغيير الوضع إلى: {BOT_MODE}")


async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING

    if BOT_RUNNING:
        await update.message.reply_text("⚠️ البوت يعمل بالفعل.")
        return

    BOT_RUNNING = True
    await update.message.reply_text(f"🚀 تشغيل نظام التداول في وضع: {BOT_MODE}")

    # تشغيل نظام التداول في Thread منفصل
    threading.Thread(
        target=RUN_CALLBACK,
        args=(BOT_MODE,),
        daemon=True
    ).start()


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING

    if not BOT_RUNNING:
        await update.message.reply_text("⚠️ البوت متوقف بالفعل.")
        return

    BOT_RUNNING = False
    await update.message.reply_text("🛑 تم إيقاف نظام التداول.")

    STOP_CALLBACK()


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 حالة البوت:\n"
        f"الوضع: {BOT_MODE}\n"
        f"يعمل: {'نعم' if BOT_RUNNING else 'لا'}"
    )


def start_telegram_bot(run_callback, stop_callback):
    global RUN_CALLBACK, STOP_CALLBACK

    RUN_CALLBACK = run_callback
    STOP_CALLBACK = stop_callback

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mode", mode))
    app.add_handler(CommandHandler("run", run_bot))
    app.add_handler(CommandHandler("stop", stop_bot))
    app.add_handler(CommandHandler("status", status))

    print("🤖 بوت تيليجرام يعمل الآن…")
    app.run_polling()
