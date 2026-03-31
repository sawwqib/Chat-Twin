import json
import re
import logging
import asyncio
import aiohttp
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
TELEGRAM_BOT_TOKEN   = "YOUR_BOT_API_KEY_HERE"
POLLINATIONS_API_KEY = "YOUR_API_KEY"
POLLINATIONS_MODEL   = "openai"  # options: openai, claude, gemini, deepseek, grok, mistral
# ─────────────────────────────────────────────

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

WAITING_FOR_FILE, WAITING_FOR_PERSON_CHOICE, CHATTING = range(3)

sessions = {}  # { user_id: { "parsed": {...}, "twin": ChatTwin } }


# ══════════════════════════════════════════════
#  PARSERS
# ══════════════════════════════════════════════

def parse_telegram(data: dict) -> dict:
    messages_by_person = {}
    for msg in data.get("messages", []):
        if msg.get("type") != "message":
            continue
        sender = msg.get("from") or msg.get("actor")
        if not sender:
            continue
        content = msg.get("text", "")
        if isinstance(content, list):
            content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
        content = content.strip()
        if content:
            messages_by_person.setdefault(sender, []).append(content)
    return {"source": "Telegram", "participants": list(messages_by_person.keys()), "messages_by_person": messages_by_person}


def parse_whatsapp(raw_text: str) -> dict:
    LINE_RE = re.compile(
        r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}),?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[aApP][mM])?\s*[-–]\s*(.+?):\s(.+)$"
    )
    SKIP = {"messages and calls are end-to-end encrypted", "missed voice call",
            "missed video call", "this message was deleted", "<media omitted>", "null"}
    messages_by_person = {}
    current_sender, current_lines = None, []

    def flush():
        if current_sender and current_lines:
            content = " ".join(current_lines).strip()
            if content.lower() not in SKIP and not content.startswith("\u200e"):
                messages_by_person.setdefault(current_sender, []).append(content)

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if m:
            flush()
            current_sender = m.group(2).strip()
            current_lines = [m.group(3).strip()]
        elif current_sender:
            current_lines.append(line)
    flush()
    return {"source": "WhatsApp", "participants": list(messages_by_person.keys()), "messages_by_person": messages_by_person}


def parse_instagram(data: dict) -> dict:
    messages_by_person = {}
    for msg in data.get("messages", []):
        sender = msg.get("sender_name", "").strip()
        content = msg.get("content", "").strip()
        if not sender or not content:
            continue
        try:
            sender = sender.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        try:
            content = content.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        messages_by_person.setdefault(sender, []).append(content)
    return {"source": "Instagram / Messenger", "participants": list(messages_by_person.keys()), "messages_by_person": messages_by_person}


def parse_discord_txt(raw_text: str) -> dict:
    HEADER_RE = re.compile(r"^(.+?)\s+[—–]\s+(?:Today|Yesterday|\d{1,2}/\d{1,2}/\d{4}) at .+$")
    messages_by_person = {}
    current_sender, current_lines = None, []

    def flush():
        if current_sender and current_lines:
            content = " ".join(current_lines).strip()
            if content:
                messages_by_person.setdefault(current_sender, []).append(content)

    for line in raw_text.split("\n"):
        m = HEADER_RE.match(line.strip())
        if m:
            flush()
            current_sender = m.group(1).strip()
            current_lines = []
        elif current_sender and line.strip():
            current_lines.append(line.strip())
    flush()
    return {"source": "Discord", "participants": list(messages_by_person.keys()), "messages_by_person": messages_by_person}


def parse_discord_json(data: list) -> dict:
    messages_by_person = {}
    for msg in data:
        author = msg.get("author", {})
        name = author.get("name") or author.get("username", "Unknown")
        content = msg.get("content", "").strip()
        if content and name:
            messages_by_person.setdefault(name, []).append(content)
    return {"source": "Discord (JSON)", "participants": list(messages_by_person.keys()), "messages_by_person": messages_by_person}


def detect_and_parse(raw_text: str, filename: str) -> dict:
    lower = filename.lower()

    if lower.endswith(".json"):
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            raise ValueError("File has .json extension but is not valid JSON.")
        if isinstance(data, list):
            return parse_discord_json(data)
        if isinstance(data, dict):
            if "messages" in data and "type" in data:
                return parse_telegram(data)
            if "messages" in data:
                return parse_instagram(data)
        raise ValueError("Unrecognized JSON format.")

    if lower.endswith(".txt"):
        wa_lines = raw_text.strip().split("\n")[:20]
        wa_pattern = re.compile(r"^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})[,\s]+\d{1,2}:\d{2}")
        if sum(1 for l in wa_lines if wa_pattern.match(l.strip())) >= 2:
            return parse_whatsapp(raw_text)
        if "— Today at" in raw_text or "— Yesterday at" in raw_text or " — " in raw_text:
            return parse_discord_txt(raw_text)
        raise ValueError("Could not detect format from .txt file.")

    raise ValueError(f"Unsupported file type: {filename}")


# ══════════════════════════════════════════════
#  CHAT TWIN  (Pollinations API)
# ══════════════════════════════════════════════

class ChatTwin:
    MAX_CHARS = 60000 * 4  # ~60k tokens worth of examples
    API_URL   = "https://gen.pollinations.ai/v1/chat/completions"

    def __init__(self, person_name: str, messages: list):
        self.person_name = person_name
        self.conversation_history = []
        self.system_prompt = self._build_system_prompt(messages)

    def _build_system_prompt(self, messages: list) -> str:
        examples, total_chars = [], 0
        for msg in reversed(messages):
            if total_chars + len(msg) > self.MAX_CHARS:
                break
            examples.append(msg)
            total_chars += len(msg)
        examples.reverse()

        examples_text = "\n".join(f"- {m}" for m in examples)
        return (
            f"You are roleplaying as {self.person_name}, a real person based on their actual chat history.\n\n"
            f"Respond EXACTLY as {self.person_name} would — matching their:\n"
            f"- Vocabulary and word choices\n"
            f"- Sentence length and structure\n"
            f"- Punctuation style (or lack thereof)\n"
            f"- Emoji usage\n"
            f"- Slang, abbreviations, expressions\n"
            f"- Overall energy and tone\n\n"
            f"Here are {len(examples)} of {self.person_name}'s real messages (out of {len(messages)} total):\n\n"
            f"{examples_text}\n\n"
            f"RULES:\n"
            f"1. NEVER break character or acknowledge you are an AI\n"
            f"2. Respond naturally as {self.person_name} — never say you 'can't' do something\n"
            f"3. Match their typical message length exactly\n"
            f"4. Mirror typos, casual spelling, and formatting if that's their style\n"
            f"5. You ARE {self.person_name}. Respond as them, not about them."
        )

    async def respond(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})

        payload = {
            "model": POLLINATIONS_MODEL,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *self.conversation_history,
            ],
            "max_tokens": 500,
        }

        headers = {
            "Authorization": f"Bearer {POLLINATIONS_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Pollinations API error {resp.status}: {text}")
                data = await resp.json()

        reply = data["choices"][0]["message"]["content"].strip()
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply


# ══════════════════════════════════════════════
#  BOT HANDLERS
# ══════════════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Welcome to Chat Twin Bot!*\n\n"
        "Send me an exported chat file and I'll let you chat with a digital twin of anyone in it.\n\n"
        "Supported formats:\n"
        "• Telegram → `.json`\n"
        "• WhatsApp → `.txt`\n"
        "• Instagram → `.json`\n"
        "• Facebook Messenger → `.json`\n"
        "• Discord → `.txt` / `.json`\n"
        "Created By @sawwqib",
        parse_mode="Markdown",
    )
    return WAITING_FOR_FILE


async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    document = update.message.document

    if not document:
        await update.message.reply_text("Please send a file.")
        return WAITING_FOR_FILE

    filename = document.file_name or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("json", "txt"):
        await update.message.reply_text("❌ Please send a `.json` or `.txt` chat export.")
        return WAITING_FOR_FILE

    await update.message.reply_text("⏳ Parsing your chat export...")

    file = await document.get_file()
    raw_bytes = await file.download_as_bytearray()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    try:
        parsed = detect_and_parse(raw_text, filename)
    except Exception as e:
        await update.message.reply_text(f"❌ Could not parse this file.\n\n`{e}`", parse_mode="Markdown")
        return WAITING_FOR_FILE

    if not parsed["participants"]:
        await update.message.reply_text("❌ No participants found. Try a different export.")
        return WAITING_FOR_FILE

    sessions[user_id] = {"parsed": parsed, "twin": None}
    participants = parsed["participants"]

    if len(participants) == 1:
        return await _start_twin(update, context, user_id, participants[0])

    keyboard = [[p] for p in participants]
    await update.message.reply_text(
        f"✅ Found *{len(participants)} people*. Who do you want to clone?",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
    )
    return WAITING_FOR_PERSON_CHOICE


async def handle_person_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    choice = update.message.text.strip()
    session = sessions.get(user_id)

    if not session:
        await update.message.reply_text("Session expired. Please send the file again.")
        return WAITING_FOR_FILE

    if choice not in session["parsed"]["participants"]:
        keyboard = [[p] for p in session["parsed"]["participants"]]
        await update.message.reply_text(
            "Please choose one of the listed names.",
            reply_markup=ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True),
        )
        return WAITING_FOR_PERSON_CHOICE

    return await _start_twin(update, context, user_id, choice)


async def _start_twin(update, context, user_id, person):
    session = sessions[user_id]
    parsed = session["parsed"]
    messages = parsed["messages_by_person"].get(person, [])

    twin = ChatTwin(person_name=person, messages=messages)
    sessions[user_id]["twin"] = twin

    await update.message.reply_text(
        f"🧠 *Twin ready!*\n\n"
        f"Person: *{person}*\n"
        f"Messages analyzed: *{len(messages)}*\n"
        f"Source: *{parsed['source']}*\n\n"
        f"Start chatting! I'll respond as *{person}*.\n"
        f"/reset to start over.",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove(),
    )
    return CHATTING


async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = sessions.get(user_id)

    if not session or not session.get("twin"):
        await update.message.reply_text("No active twin. Send /start to begin.")
        return CHATTING

    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        reply = await session["twin"].respond(update.message.text)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Twin error: {e}")
        await update.message.reply_text("⚠️ Something went wrong. Try again.")

    return CHATTING


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.pop(update.effective_user.id, None)
    await update.message.reply_text("🔄 Reset. Send a new chat export to start.", reply_markup=ReplyKeyboardRemove())
    return WAITING_FOR_FILE


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sessions.pop(update.effective_user.id, None)
    await update.message.reply_text("👋 Bye! Send /start to begin a new session.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

async def async_main():
    """Async main function to properly handle the event loop"""
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            WAITING_FOR_FILE: [
                MessageHandler(filters.Document.ALL, handle_file),
                MessageHandler(filters.TEXT & ~filters.COMMAND, lambda u, c: u.message.reply_text("Please send a chat export file.")),
            ],
            WAITING_FOR_PERSON_CHOICE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_person_choice),
            ],
            CHATTING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat),
                CommandHandler("reset", reset),
            ],
        },
        fallbacks=[CommandHandler("stop", stop)],
        allow_reentry=True,
    )

    app.add_handler(conv)
    logger.info("Bot is running...")
    
    # Initialize and start the application
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    
    # Keep the bot running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


def main():
    """Main entry point with proper asyncio handling"""
    try:
        # For Python 3.10+, we need to create and set the event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(async_main())
        finally:
            loop.close()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")


if __name__ == "__main__":
    main()