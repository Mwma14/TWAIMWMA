# twai.py - v5 Professional Edition

import os
import logging
import asyncio
import base64
import httpx
import io
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update, constants, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
    ConversationHandler
)
from telegram.error import TimedOut, BadRequest

# --- 1. INITIAL SETUP & CONFIGURATION ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

if not all([TELEGRAM_TOKEN, GOOGLE_API_KEY, STABILITY_API_KEY]):
    raise ValueError("All API keys must be set in the .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

genai.configure(api_key=GOOGLE_API_KEY)
text_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- 2. PROFESSIONAL UI ELEMENTS ---
GET_IMAGE_PROMPT = range(1)

# ### UPDATED: Professional button text (no slashes) ###
main_menu_keyboard = [["Image ✨", "Clear 🧹"], ["Help ❓", "Menu 📋"]]
main_menu_markup = ReplyKeyboardMarkup(main_menu_keyboard, resize_keyboard=True)

# ### NEW: Dedicated cancel button for conversations ###
cancel_keyboard = [["Cancel ❌"]]
cancel_markup = ReplyKeyboardMarkup(cancel_keyboard, resize_keyboard=True, one_time_keyboard=True)

spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# --- 3. COMMAND HANDLERS ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends the consolidated, professional welcome message."""
    user_name = update.effective_user.first_name
    welcome_message = (
        f"Hello, {user_name}! Welcome to your AI assistant.\n\n"
        "**Here's what I can do:**\n\n"
        "💬 **Chat:** Just type any message to talk to me.\n"
        "✨ **Image:** Create a unique image from a text prompt.\n"
        "🧹 **Clear:** Start a fresh text conversation.\n"
        "📋 **Menu:** Bring back this keyboard menu.\n"
        "❓ **Help:** Show this message again."
    )
    await update.message.reply_text(welcome_message, parse_mode=constants.ParseMode.MARKDOWN, reply_markup=main_menu_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Alias for the start command to show the help text again."""
    await start_command(update, context)

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Here is the main menu:", reply_markup=main_menu_markup)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if 'history' in context.user_data:
        del context.user_data['history']
        await update.message.reply_text("Our text conversation history has been cleared.")
    else:
        await update.message.reply_text("There's no conversation history to clear.")
    
# --- 4. GUIDED IMAGE GENERATION CONVERSATION ---

async def image_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Starts image generation and shows the cancel button."""
    await update.message.reply_text(
        "Please describe the image you want to create.",
        reply_markup=cancel_markup
    )
    return GET_IMAGE_PROMPT

async def generate_image_from_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    prompt = update.message.text
    logger.info(f"Image request from user {user_id} with prompt: '{prompt}'")
    status_message = await update.message.reply_text("`Initializing AI...`", parse_mode=constants.ParseMode.MARKDOWN)
    
    generation_task = asyncio.create_task(_image_api_call(prompt))
    context.user_data['active_task'] = generation_task

    i = 0
    while not generation_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(generation_task), timeout=0.3)
        except asyncio.TimeoutError:
            new_text = f"`Processing... {spinner_frames[i % len(spinner_frames)]}`"
            try:
                await context.bot.edit_message_text(text=new_text, chat_id=status_message.chat_id, message_id=status_message.message_id, parse_mode=constants.ParseMode.MARKDOWN)
            except BadRequest: pass
            i += 1
        except asyncio.CancelledError:
            await context.bot.edit_message_text("`🚫 Generation cancelled.`", parse_mode=constants.ParseMode.MARKDOWN, chat_id=status_message.chat_id, message_id=status_message.message_id)
            logger.info(f"Image generation cancelled by user {user_id}")
            await menu_command(update, context)
            return ConversationHandler.END

    image_bytes = generation_task.result()
    del context.user_data['active_task']
    await context.bot.delete_message(chat_id=status_message.chat_id, message_id=status_message.message_id)

    if image_bytes:
        try:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=image_bytes,
                caption=f"Image generated for:\n*'{prompt}'*", parse_mode=constants.ParseMode.MARKDOWN
            )
        except TimedOut:
            await update.message.reply_text("The upload to Telegram timed out. Please try again.")
        except Exception as e:
            await update.message.reply_text("An error occurred while sending the photo.")
    else:
        await update.message.reply_text("Sorry, image generation failed. The prompt may have been rejected or the service may be unavailable.")
    
    await menu_command(update, context)
    return ConversationHandler.END

async def _image_api_call(prompt: str) -> io.BytesIO | None:
    # ### UPDATED: Switched to the universally available, high-quality SDXL model ###
    api_host = 'https://api.stability.ai'
    engine_id = "stable-diffusion-xl-1024-v1-0"
    api_url = f"{api_host}/v1/generation/{engine_id}/text-to-image"

    headers = { "Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json" }
    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1,
        "style_preset": "photographic"
    }
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            image_b64 = data["artifacts"][0]["base64"]
            return io.BytesIO(base64.b64decode(image_b64))
    except (httpx.RequestError, KeyError, IndexError) as e:
        logger.error(f"Stability AI API call failed: {e}")
        return None

async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handles both /cancel command and 'Cancel ❌' button press."""
    if 'active_task' in context.user_data:
        context.user_data['active_task'].cancel()
    await update.message.reply_text("Action cancelled.", reply_markup=main_menu_markup)
    return ConversationHandler.END

# --- 5. CORE TEXT MESSAGE HANDLING ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This function remains the same
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
    if 'history' not in context.user_data: context.user_data['history'] = []
    try:
        chat = text_model.start_chat(history=context.user_data['history'])
        response = await chat.send_message_async(update.message.text)
        context.user_data['history'] = chat.history
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("Sorry, an error occurred with the text generation.")

# --- 6. ERROR HANDLING & BOT LAUNCH ---
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).read_timeout(60).write_timeout(60).build()
    
    # ### UPDATED: ConversationHandler with robust cancel and professional buttons ###
    image_conv_handler = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex("^Image ✨$"), image_entry)],
        states={
            GET_IMAGE_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, generate_image_from_prompt)]
        },
        fallbacks=[
            CommandHandler("cancel", cancel_command),
            MessageHandler(filters.Regex("^Cancel ❌$"), cancel_command)
        ],
    )
    
    app.add_handler(image_conv_handler)
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Regex("^Help ❓$"), help_command))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(MessageHandler(filters.Regex("^Menu 📋$"), menu_command))
    app.add_handler(CommandHandler("clear", clear_command))
    app.add_handler(MessageHandler(filters.Regex("^Clear 🧹$"), clear_command))
    
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("Bot v5 Professional Edition is starting to poll...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()