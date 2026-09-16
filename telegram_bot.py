"""Optional Telegram control interface."""

from __future__ import annotations

import logging
import os
import threading

try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
except ImportError:  # Optional dependency: research mode must not need it.
    Update = object
    ApplicationBuilder = None
    CommandHandler = None
    ContextTypes = object

# حالة البوت
BOT_MODE = "DEMO"     # أو LIVE
BOT_RUNNING = False
LIVE_CONFIRMED = False
ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID")
logger = logging.getLogger(__name__)

# كولباكات يتم حقنها من main.py
RUN_CALLBACK = None
STOP_CALLBACK = None
STATUS_CALLBACK = None

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
        "/confirm_live – تأكيد يدوي قبل الوضع الحقيقي\n"
        "/run – تشغيل البوت\n"
        "/stop – إيقاف البوت\n"
        "/status – حالة البوت\n"
        "/equity – رأس المال\n"
        "/positions – الصفقات المفتوحة"
    )


async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_MODE, LIVE_CONFIRMED

    if not context.args:
        await update.message.reply_text(f"الوضع الحالي: {BOT_MODE}")
        return

    m = context.args[0].lower()

    if m == "demo":
        BOT_MODE = "DEMO"
    elif m == "live":
        BOT_MODE = "LIVE"
        LIVE_CONFIRMED = False
    else:
        await update.message.reply_text("❌ الوضع غير معروف. استخدم demo أو live")
        return

    await update.message.reply_text(f"✅ تم تغيير الوضع إلى: {BOT_MODE}")


def _is_authorized(update: Update) -> bool:
    return not ADMIN_CHAT_ID or str(update.effective_chat.id) == ADMIN_CHAT_ID


async def confirm_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global LIVE_CONFIRMED
    if not _is_authorized(update):
        await update.message.reply_text("❌ هذا الأمر غير مصرح به.")
        return
    if BOT_MODE != "LIVE":
        await update.message.reply_text("غيّر الوضع إلى live أولًا.")
        return
    LIVE_CONFIRMED = True
    await update.message.reply_text("⚠️ تم تأكيد الوضع LIVE يدويًا.")


async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING

    if BOT_RUNNING:
        await update.message.reply_text("⚠️ البوت يعمل بالفعل.")
        return
    if BOT_MODE == "LIVE" and not LIVE_CONFIRMED:
        await update.message.reply_text("❌ أكد الوضع الحقيقي أولًا باستخدام /confirm_live.")
        return

    BOT_RUNNING = True
    await update.message.reply_text(f"🚀 تشغيل نظام التداول في وضع: {BOT_MODE}")

    # تشغيل نظام التداول في Thread منفصل
    if RUN_CALLBACK is None:
        BOT_RUNNING = False
        await update.message.reply_text("❌ لم يتم إعداد دالة التشغيل.")
        return

    threading.Thread(
        target=RUN_CALLBACK, args=(BOT_MODE,), daemon=True
    ).start()


async def stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BOT_RUNNING

    if not BOT_RUNNING:
        await update.message.reply_text("⚠️ البوت متوقف بالفعل.")
        return

    BOT_RUNNING = False
    await update.message.reply_text("🛑 تم إيقاف نظام التداول.")

    if STOP_CALLBACK is not None:
        STOP_CALLBACK()


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 حالة البوت:\n"
        f"الوضع: {BOT_MODE}\n"
        f"يعمل: {'نعم' if BOT_RUNNING else 'لا'}"
    )


async def equity(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_data = STATUS_CALLBACK() if STATUS_CALLBACK else {}
    await update.message.reply_text(f"رأس المال: {status_data.get('equity', 'غير متاح')}")


async def positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_data = STATUS_CALLBACK() if STATUS_CALLBACK else {}
    open_positions = status_data.get("positions", [])
    await update.message.reply_text(f"الصفقات المفتوحة: {len(open_positions)}")


def start_telegram_bot(run_callback, stop_callback, status_callback=None):
    global RUN_CALLBACK, STOP_CALLBACK, STATUS_CALLBACK

    RUN_CALLBACK = run_callback
    STOP_CALLBACK = stop_callback
    STATUS_CALLBACK = status_callback

    if ApplicationBuilder is None:
        raise RuntimeError(
            "Telegram control requires the optional python-telegram-bot package."
        )
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required for Telegram control.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("mode", mode))
    app.add_handler(CommandHandler("confirm_live", confirm_live))
    app.add_handler(CommandHandler("run", run_bot))
    app.add_handler(CommandHandler("stop", stop_bot))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("equity", equity))
    app.add_handler(CommandHandler("positions", positions))

    logger.info("بوت تيليجرام يعمل الآن")
    app.run_polling()
