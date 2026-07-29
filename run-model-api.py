import os
import gc
import json
import re
import glob
import asyncio
from datetime import datetime
from threading import Thread
from typing import AsyncGenerator

import torch
import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextIteratorStreamer
)

# Prevent CUDA Memory Fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CHAT_DIR = "./chat_history"
os.makedirs(CHAT_DIR, exist_ok=True)

# --- 1. Memory Check & Setup ---
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
model_path = r"D:\dev\LLM-Experiments\model3"

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

# --- 3. Helper Functions ---
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
    if not content:
        return ""
    if isinstance(content, list):
        extracted = []
        for item in content:
            if isinstance(item, dict):
                extracted.append(str(item.get("text", "")))
            else:
                extracted.append(str(item))
        return " ".join(extracted).strip()
    elif isinstance(content, dict):
        return str(content.get("text", "")).strip()
    return str(content).strip()

def compile_all_user_history():
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
        
    return (
        "Here is a chronological list of ALL user prompts from my previous chat sessions:\n\n"
        + "\n\n".join(all_user_inputs)
        + "\n\n--- INSTRUCTION ---\n"
        "Please provide a comprehensive summary of all the topics, code requests, and key concepts I have asked about across these past conversations."
    )

# --- 4. FastAPI Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StreamRequest(BaseModel):
    session_id: str
    messages: list

class LoadRequest(BaseModel):
    session_id: str

@app.get("/api/sessions")
def get_session_list():
    files = glob.glob(os.path.join(CHAT_DIR, "*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    file_names = [os.path.basename(f) for f in files]
    return {"sessions": file_names if file_names else ["No Saved Chats"]}

@app.post("/api/load_session")
def load_session(req: LoadRequest):
    if not req.session_id or req.session_id == "No Saved Chats":
        return {"history": []}
    path = os.path.join(CHAT_DIR, req.session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {"history": json.load(f)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"history": []}

@app.get("/api/create_session")
def new_session():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"session_id": f"chat_{timestamp}.json"}

@app.get("/api/summarize_prompt")
def get_summarize_prompt():
    prompt = compile_all_user_history()
    return {"prompt": prompt}

@app.post("/api/clear_vram")
def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"status": "VRAM cleared", "new_session_id": f"chat_{timestamp}.json"}

@app.post("/api/stream")
async def stream_chat(req: StreamRequest):
    MAX_HISTORY_TURNS = 10
    recent_history = req.messages[-MAX_HISTORY_TURNS:]

    clean_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in recent_history:
        role = str(msg.get("role", "user"))
        raw_text = extract_text_from_content(msg.get("content", ""))
        content = clean_text_payload(raw_text)
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

    async def event_generator() -> AsyncGenerator[str, None]:
        partial_text = ""
        try:
            for new_text in streamer:
                partial_text += new_text
                yield json.dumps({"token": new_text})
                await asyncio.sleep(0.001)

            # Save full conversation to disk on completion
            save_path = os.path.join(CHAT_DIR, req.session_id)
            full_history = req.messages + [{"role": "assistant", "content": clean_text_payload(partial_text)}]
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(full_history, f, indent=2, ensure_ascii=False)

        except Exception as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                yield json.dumps({"token": "\n⚠️ **CUDA Out of Memory Error!** Click 'Clear Conversation & Free VRAM' below."})

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)