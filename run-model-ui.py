import os
import gc
import json
import re
import glob
from datetime import datetime
import torch
import psutil
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)
from threading import Thread
import gradio as gr

# Prevent CUDA Memory Fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Storage directory for local JSON session files
CHAT_DIR = "./chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)

# --- 1. Memory Check & PyTorch Setup ---
print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())

max_memory = 10 * 1024 * 1024 * 1024  # 10GB
def check_memory_usage():
    mem_info = psutil.virtual_memory()
    if mem_info.used >= max_memory:
        print("System memory usage high! Consider closing background applications.")

check_memory_usage()

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, 0)

# --- 2. Model & Tokenizer Initialization ---
model_path = r"D:\dev\LLM-Experiments\modelcode\Qwen2.5-Coder-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print(f"\n[1/2] Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"[2/2] Loading model into VRAM...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    quantization_config=quantization_config, 
    device_map="auto",
    attn_implementation="sdpa"
)

SYSTEM_PROMPT = "You are a pro in all fields especially in coding, a helpful AI assistant."

# --- 3. Chat Storage Helpers ---
def get_session_list():
    files = glob.glob(os.path.join(CHAT_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    file_names = [os.path.basename(f) for f in files]
    return file_names if file_names else ["No Saved Chats"]

def load_session_file(filename):
    if not filename or filename == "No Saved Chats":
        return []
    path = os.path.join(CHAT_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading session file: {e}")
    return []

def save_session_file(session_id, history):
    if not session_id or not history:
        return
    path = os.path.join(CHAT_DIR, session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving session file: {e}")

def create_new_session():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"chat_{timestamp}.json"

def clean_text_payload(text):
    if not isinstance(text, str):
        return str(text)
    
    text_str = text.strip()
    if (text_str.startswith("{") and text_str.endswith("}")) or " 'text': " in text_str or ' "text": ' in text_str:
        try:
            data = json.loads(text_str)
            if isinstance(data, dict) and "text" in data:
                return clean_text_payload(data["text"])
        except Exception:
            pass
        
        match = re.search(r"['\"]text['\"]\s*:\s*['\"](.*?)['\"]\s*\}?$", text_str, re.DOTALL)
        if match:
            extracted = match.group(1)
            return extracted.replace(r"\n", "\n").replace(r"\'", "'").replace(r'\"', '"')

    return text

def extract_text_from_content(content):
    """ Extract raw text regardless of whether content is a string, list, or dict """
    if not content:
        return ""
    
    # Handle list of items (e.g. [{"type": "text", "text": "..."}])
    if isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict):
                extracted.append(str(item.get("text", "")))
            else:
                extracted.append(str(item))
        return " ".join(extracted).strip()
    
    # Handle dictionary (e.g. {"text": "..."})
    elif isinstance(content, dict):
        return str(content.get("text", "")).strip()
    
    # Handle regular string
    return str(content).strip()


def compile_all_user_history():
    """ Reads all saved JSON chat files and extracts ONLY user messages """
    files = glob.glob(os.path.join(CHAT_DIR, "*.json"))
    files.sort(key=os.path.getmtime)
    all_user_inputs = []
    
    for fpath in files:
        filename = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                user_msgs = []
                for msg in data:
                    if msg.get("role") == "user":
                        # Safely extract text from string, list, or dict formats
                        raw_text = extract_text_from_content(msg.get("content"))
                        clean_text = clean_text_payload(raw_text)
                        
                        if clean_text.strip():
                            user_msgs.append(clean_text)
                
                if user_msgs:
                    formatted_msgs = "\n".join(f"- {m}" for m in user_msgs)
                    all_user_inputs.append(f"📁 **Session ({filename})**:\n{formatted_msgs}")
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    if not all_user_inputs:
        return None
        
    compiled_prompt = (
        "Here is a chronological list of ALL user prompts from my previous chat sessions:\n\n"
        + "\n\n".join(all_user_inputs)
        + "\n\n--- INSTRUCTION ---\n"
        "Please provide a comprehensive summary of all the topics, code requests, and key concepts I have asked about across these past conversations."
    )
    return compiled_prompt

# --- 4. Generation Logic ---
def user_submit(user_message, history, current_session_id):
    if not user_message.strip():
        return "", history, current_session_id
    
    if history is None:
        history = []
        
    if not current_session_id:
        current_session_id = create_new_session()
        
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})
    
    return "", history, current_session_id

def trigger_all_history_summary():
    """ Starts a fresh session loaded with all past user prompts for the model to summarize """
    compiled_prompt = compile_all_user_history()
    new_id = create_new_session()
    
    if not compiled_prompt:
        history = [
            {"role": "user", "content": "Summarize all my past chats."},
            {"role": "assistant", "content": "⚠️ No previous chat history found in `./chat_history` to summarize."}
        ]
        return history, new_id
    
    history = [
        {"role": "user", "content": compiled_prompt},
        {"role": "assistant", "content": ""}
    ]
    return history, new_id

def bot_response(history, current_session_id):
    if not history or len(history) < 2:
        yield history, gr.update()
        return

    MAX_HISTORY_TURNS = 10
    recent_history = history[:-1][-MAX_HISTORY_TURNS:]

    raw_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent_history

    clean_messages = []
    for msg in raw_messages:
        role = str(msg.get("role", "user"))
        content = clean_text_payload(str(msg.get("content", "")))
        if content.strip() or role == "system":
            clean_messages.append({"role": role, "content": content})

    formatted_prompt = tokenizer.apply_chat_template(
        clean_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    try:
        for new_text in streamer:
            partial_text += new_text
            history[-1]["content"] = clean_text_payload(partial_text)
            yield history, gr.update()
            
        save_session_file(current_session_id, history)
        yield history, gr.update(choices=get_session_list(), value=current_session_id)

    except Exception as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            history[-1]["content"] = "⚠️ **CUDA Out of Memory Error!** Click 'Clear Conversation & Free VRAM' below."
            yield history, gr.update()
        else:
            raise e

def switch_chat_session(selected_file):
    history = load_session_file(selected_file)
    return history, selected_file

def start_new_chat():
    new_id = create_new_session()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return [], new_id

def clear_vram_and_history():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    new_id = create_new_session()
    return [], new_id

# --- 5. Theme & CSS Styling ---
force_light_js = """
() => {
    document.body.classList.remove('dark');
    document.documentElement.classList.remove('dark');
    const url = new URL(window.location.href);
    if (!url.searchParams.has('__theme')) {
        url.searchParams.set('__theme', 'light');
        window.history.replaceState({}, '', url.href);
    }
}
"""

clean_light_css = """
:root, body, .gradio-container, .dark {
    --body-background-fill: #f4f4f5 !important;
    --body-text-color: #18181b !important;
    --body-text-color-subdued: #71717a !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e4e4e7 !important;
    --block-border-width: 1px !important;
    --panel-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fafafa !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #d4d4d8 !important;
    --input-text-color: #18181b !important;
}

.chatbot, .chatbot > .wrapper, div[aria-label="chatbot"] {
    background-color: #ffffff !important;
    color: #18181b !important;
}

.chatbot .message, .chatbot .message-content, .chatbot .prose {
    background-color: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
}

.chatbot code, .gradio-container code {
    background-color: #f4f4f5 !important;
    color: #09090b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

.chatbot pre, .gradio-container pre, .chatbot pre code {
    background-color: #f4f4f5 !important;
    color: #09090b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

# --- 6. Interface Layout ---
with gr.Blocks(title="Qwen 2.5 Local Assistant") as demo:
    current_session = gr.State(value=create_new_session())

    gr.Markdown(
        """
        # 🤖 Qwen2.5-Coder-7B Local Assistant
        *Running locally via PyTorch & BitsAndBytes 4-Bit Quantization*
        """
    )
    
    # Session Controls Bar
    with gr.Row():
        new_chat_btn = gr.Button("➕ New Chat", variant="primary", scale=2)
        chat_selector = gr.Dropdown(
            choices=get_session_list(),
            value=get_session_list()[0] if get_session_list() else None,
            show_label=False,
            container=False,
            interactive=True,
            scale=5
        )
        load_chat_btn = gr.Button("📥 Load Chat", variant="secondary", scale=2)
        summarize_all_btn = gr.Button("📊 Summarize All Past Chats", variant="secondary", scale=3)
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)

    chatbot = gr.Chatbot(
        label="Conversation History",
        height=600,
        render_markdown=True
    )
    
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Type a prompt or paste large code blocks/context here... (Shift+Enter for new line)",
            lines=3,
            max_lines=15,
            show_label=False,
            scale=8
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Conversation & Free VRAM", variant="secondary")

    # Wire Send / Submit
    msg_input.submit(
        user_submit, 
        inputs=[msg_input, chatbot, current_session], 
        outputs=[msg_input, chatbot, current_session]
    ).then(
        bot_response, 
        inputs=[chatbot, current_session], 
        outputs=[chatbot, chat_selector]
    )

    submit_btn.click(
        user_submit, 
        inputs=[msg_input, chatbot, current_session], 
        outputs=[msg_input, chatbot, current_session]
    ).then(
        bot_response, 
        inputs=[chatbot, current_session], 
        outputs=[chatbot, chat_selector]
    )

    # Wire Session Buttons
    new_chat_btn.click(start_new_chat, outputs=[chatbot, current_session])
    load_chat_btn.click(switch_chat_session, inputs=[chat_selector], outputs=[chatbot, current_session])
    chat_selector.change(switch_chat_session, inputs=[chat_selector], outputs=[chatbot, current_session])
    refresh_btn.click(lambda: gr.update(choices=get_session_list()), outputs=[chat_selector])
    clear_btn.click(clear_vram_and_history, outputs=[chatbot, current_session])

    # Wire Summarize All Past Chats
    summarize_all_btn.click(
        trigger_all_history_summary,
        outputs=[chatbot, current_session]
    ).then(
        bot_response,
        inputs=[chatbot, current_session],
        outputs=[chatbot, chat_selector]
    )

if __name__ == "__main__":
    print("\n--- Model loaded! Launching web server... ---")
    demo.queue().launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        inbrowser=True,
        theme=theme,
        css=clean_light_css,
        js=force_light_js
    )