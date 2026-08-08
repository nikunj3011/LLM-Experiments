from datetime import datetime
import glob
import io
import os
import gc
import json
import time
import asyncio
from pathlib import Path
import re
import base64
import mimetypes
from typing import Optional, List, Dict, Any
from threading import Thread
import uuid
import logging
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from fastapi.staticfiles import StaticFiles
from PIL import Image
import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import torch
import shutil
import subprocess
import fitz           
import docx
import pandas as pd
import scipy.io.wavfile

import urllib.parse
import requests
from bs4 import BeautifulSoup
import torchaudio

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    ChameleonForConditionalGeneration,
    MusicgenForConditionalGeneration,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

from llama_cpp import Llama
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("uvicorn.error")

# ------------------------------------------------------------------------------
# APP SETUP & CORS
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Dynamic VRAM Chat & Any-to-Any Vision API",
    version="4.5",
    description="Optimized dynamic VRAM switcher with multi-backend streaming"
)

os.makedirs("static", exist_ok=True)
app.mount("/temp_uploads", StaticFiles(directory="temp_uploads"), name="temp_uploads")

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_upload_size:
                raise HTTPException(status_code=413, detail="File too large")
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=50 * 1024 * 1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS_DIR = "./chat_history"
TEMP_UPLOADS_DIR = "./temp_uploads"
SYSTEM_PROMPT = "You are a pro in all fields especially in coding, an AI assistant."
CONFIG_PATH = Path("config.json")
N_CTX = 5120
# file_path = os.path.join("./system_prompts/Anthropic/Official/", "2026-06-09-claude-fable-5.md")
file_path = Path("system_prompt.md")
try:
    # Open and read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        SYSTEM_PROMPT = SYSTEM_PROMPT + file.read().strip()
except FileNotFoundError:
    # Fallback if the file is missing
    SYSTEM_PROMPT = "You are a pro in all fields especially in coding, an AI assistant."
    print(f"Warning: {file_path} not found. Using default prompt.")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

def load_models_from_config(config_file: Path) -> List[Dict[str, Any]]:
    """Loads the available models list from a JSON configuration file."""
    if not config_file.exists():
        # Fallback or error handling if the file is missing
        print(f"Warning: {config_file} not found. Returning empty list.")
        return []

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("AVAILABLE_MODELS", [])
    
# ------------------------------------------------------------------------------
# DYNAMIC MODEL REGISTRY
# ------------------------------------------------------------------------------
AVAILABLE_MODELS: List[Dict[str, Any]] = load_models_from_config(CONFIG_PATH)


def get_model_config(model_id: str) -> Dict[str, Any]:
    selected_id = (model_id or "qwen").lower().strip()
    for m in AVAILABLE_MODELS:
        if m["id"].lower() == selected_id:
            return m
    return AVAILABLE_MODELS[0]

def approx_truncate_user_prompt(user_prompt: str, max_gen: int = 512) -> str:
    # 1 token ≈ 3.5 characters (safe/conservative threshold)
    CHARS_PER_TOKEN = 3.5 
    
    # Calculate non-user character usage
    overhead_chars = len(SYSTEM_PROMPT) + (max_gen * 4) + 250  # 250 char buffer
    max_user_chars = int((N_CTX * CHARS_PER_TOKEN) - overhead_chars)
    
    if len(user_prompt) > max_user_chars:
        print(f"[Truncating] User prompt reduced from {len(user_prompt)} to {max_user_chars} chars.")
        return user_prompt[:max_user_chars]
        
    return user_prompt

# ==============================================================================
# DYNAMIC VRAM MODEL MANAGER
# ==============================================================================
class DynamicModelManager:
    """Manages loading and unloading of models into VRAM to prevent OOM errors."""
    def __init__(self):
        self.active_model_id: Optional[str] = None
        self.active_config: Optional[Dict[str, Any]] = None
        self.model = None
        self.tokenizer_or_processor = None
        self.lock = asyncio.Lock()

    def unload_vram(self):
        """Forces the current model out of memory and runs garbage collection."""
        if self.model is not None:
            print(f"[VRAM Manager] Unloading '{self.active_model_id}' from VRAM...")
            del self.model
            del self.tokenizer_or_processor
            self.model = None
            self.tokenizer_or_processor = None
            self.active_model_id = None
            self.active_config = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[VRAM Manager] VRAM cleared successfully.")

    async def load_model_by_config(self, model_id: str):
        """Loads a model dynamically based on its defined backend type."""
        config = get_model_config(model_id)
        target_id = config["id"]
        backend_type = config["backend_type"]

        async with self.lock:
            # Skip if already loaded
            if self.active_model_id == target_id:
                return self.model, self.tokenizer_or_processor, config

            print(f"[VRAM Manager] Requesting '{target_id}' ({backend_type.upper()}). Swapping VRAM...")
            self.unload_vram() # Unload previous before loading new

            try:
                # Backend 1: GGUF (llama.cpp)
                if backend_type == "gguf":
                    mmproj_path = config.get("mmproj_path")
                    valid_clip_path = mmproj_path if (mmproj_path and os.path.exists(mmproj_path)) else None
                    chat_handler = Qwen25VLChatHandler(clip_model_path=valid_clip_path) if valid_clip_path else None
                    chat_format = "chatml" if "qwen" in target_id or "gemma" in target_id else None
                    
                    self.model = Llama(
                        model_path=config["path"],
                        clip_model_path=valid_clip_path,
                        chat_format=chat_format,
                        chat_handler=chat_handler,
                        n_gpu_layers=20,  
                        n_ctx=5000,
                        n_threads=6,
                        use_mmap=True,
                    )
                    self.tokenizer_or_processor = None

                # Backend 2: MusicGen
                elif backend_type == "musicgen":
                    model_path = config["path"]
                    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    self.tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, local_files_only=os.path.exists(model_path))
                    self.model = MusicgenForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=compute_dtype, local_files_only=os.path.exists(model_path)
                    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

                # Backend 3: Safetensors (Transformers)
                elif backend_type == "safetensors":
                    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype
                    )
                    model_path = config["path"]

                    if config.get("supports_vision"):
                        self.tokenizer_or_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=os.path.exists(model_path))
                        if config.get("model_type") == "chameleon":
                            self.model = ChameleonForConditionalGeneration.from_pretrained(
                                model_path, quantization_config=bnb_config, device_map="auto", torch_dtype=compute_dtype, trust_remote_code=True, local_files_only=os.path.exists(model_path)
                            ).eval()
                    else:
                        self.tokenizer_or_processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=os.path.exists(model_path))
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path, quantization_config=bnb_config, device_map="auto", trust_remote_code=True, local_files_only=os.path.exists(model_path)
                        ).eval()
                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")

                self.active_model_id = target_id
                self.active_config = config
                return self.model, self.tokenizer_or_processor, config

            except Exception as e:
                logger.error(f"[VRAM Manager] Failed to load model '{target_id}': {e}", exc_info=True)
                self.unload_vram()
                raise

manager = DynamicModelManager()

# ==============================================================================
# SCHEMAS & UTILITIES
# ==============================================================================
class LoadRequest(BaseModel):
    session_id: str

# ------------------------------------------------------------------------------
# SCHEMAS
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any
    filePreview: Optional[str] = None
    fileName: Optional[str] = None
    modelUsed: Optional[str] = None

class LoadRequest(BaseModel):
    session_id: str

# ------------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------------
def cleanup_temp_uploads():
    now = time.time()
    # for f in os.listdir(TEMP_UPLOADS_DIR):
    #     fpath = os.path.join(TEMP_UPLOADS_DIR, f)
    #     if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 1800:
    #         try:
    #             os.remove(fpath)
    #         except Exception:
    #             pass

def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extracts evenly spaced frames from a video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return frames

    interval = max(1, total_frames // max_frames)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames

def extract_text_from_content(content: Any) -> str:
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

def fit_messages_to_context(
    messages: list[dict],
    tokenizer_or_model,
    max_context_limit: int = 4096,
    max_generation_tokens: int = 2048,
    safety_buffer: int = 64
) -> list[dict]:
    """Prunes older chat history messages to ensure total context fits within limits."""
    budget = max_context_limit - max_generation_tokens - safety_buffer
    
    # Helper to count tokens roughly or precisely
    def count_tokens(msg_list: list[dict]) -> int:
        full_text = "".join([m.get("content", "") if isinstance(m.get("content"), str) else "" for m in msg_list])
        if hasattr(tokenizer_or_model, "tokenize"):
            # llama-cpp or GGUF tokenizer
            return len(tokenizer_or_model.tokenize(full_text.encode("utf-8")))
        elif hasattr(tokenizer_or_model, "encode"):
            # HF Tokenizer
            return len(tokenizer_or_model.encode(full_text))
        return int(len(full_text) / 3.5) # Conservative fallback ratio

    # Keep system message separate if present
    system_msg = [m for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]

    # Prune oldest messages first until we are under budget
    while chat_msgs and count_tokens(system_msg + chat_msgs) > budget:
        # Don't delete the last remaining user prompt
        if len(chat_msgs) <= 1:
            # If even the last user message exceeds budget, truncate its text content
            text_content = chat_msgs[-1]["content"]
            if isinstance(text_content, str):
                max_chars = int(budget * 3.5)
                chat_msgs[-1]["content"] = text_content[-max_chars:] # keep recent portion
            break
        # Pop the oldest turn (User + Assistant pair or single message)
        chat_msgs.pop(0)

    return system_msg + chat_msgs

def append_and_save_chat(
    session_id: Optional[str],
    user_msg: str,
    assistant_msg: str,
    model_used: str,
    file_preview: Optional[str] = None,
    file_name: Optional[str] = None
):
    if not session_id:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_id = f"chat_{timestamp}.json"
    
    filename = session_id if session_id.endswith(".json") else f"{session_id}.json"
    filepath = os.path.join(SESSIONS_DIR, os.path.basename(filename))

    messages = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                messages = data.get("messages", data.get("history", []))
        except Exception as e:
            logger.error(f"Error reading session file {filepath}: {e}")

    user_entry = {"role": "user", "content": user_msg}
    if file_preview:
        user_entry["filePreview"] = file_preview
    if file_name:
        user_entry["fileName"] = file_name

    assistant_entry = {
        "role": "assistant",
        "content": assistant_msg,
        "modelUsed": model_used
    }

    messages.extend([user_entry, assistant_entry])

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving session file {filepath}: {e}")
    
    return session_id

async def process_uploaded_file(file_path: str):
    ext = Path(file_path).suffix.lower()
    result = {"type": "file", "path": file_path, "text": "", "media": None}

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        result["media"] = Image.open(file_path)
        result["type"] = "image"
        return result
    elif ext == ".pdf":
        doc = fitz.open(file_path)
        result["text"] = "".join([page.get_text() for page in doc])
        return result
    elif ext == ".docx":
        document = docx.Document(file_path)
        result["text"] = "\n".join(p.text for p in document.paragraphs)
        return result
    elif ext in [".xlsx", ".xls"]:
        sheets = pd.ExcelFile(file_path)
        output = ""
        for sheet in sheets.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet)
            output += f"\nSheet: {sheet}\n" + df.to_string()
        result["text"] = output
        return result
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        result["text"] = df.to_string()
        return result
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            result["text"] = json.dumps(json.load(f), indent=2)
        return result
    elif ext in [".txt", ".py", ".js", ".ts", ".java", ".cpp", ".cs", ".html", ".css", ".xml", ".yaml", ".yml", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            result["text"] = f.read()
        return result
    else:
        result["text"] = f"Uploaded file type {ext} cannot be parsed as text."
        return result

def sanitize_image(image_path: str) -> None:
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(image_path, format="JPEG", quality=95)
    except Exception as e:
        logger.error(f"Failed to sanitize image metadata: {e}")
from duckduckgo_search import DDGS

def perform_web_search(query: str, max_results: int = 10) -> str:
    try:
        # Use context manager & specify fallback backend and explicit region
        with DDGS() as ddgs:
            # Try 'html' or 'lite' backend if default returns []
            results = list(ddgs.text(
                query, 
                max_results=max_results, 
                backend="html",      # Options: 'api', 'html', 'lite'
                region="wt-wt"       # 'wt-wt' stands for Worldwide
            ))
            
            # Fallback to 'lite' if 'html' yields nothing
            if not results:
                results = list(ddgs.text(query, max_results=max_results, backend="lite"))

            if not results:
                return ""
            
            search_summary = "\n--- Live Web Search Results ---\n"
            for idx, item in enumerate(results, 1):
                search_summary += f"[{idx}] {item['title']}\nSnippet: {item['body']}\nURL: {item['href']}\n\n"
            search_summary += "--- End of Web Search Results ---\n"
            return search_summary

    except Exception as e:
        print(f"[WebSearch Error]: {e}")
        return ""

def perform_google_scrape(query: str, max_results: int = 3) -> str:
    """Scrapes Google Search directly without official API keys or external search libraries."""
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}&num={max_results + 2}&hl=en"

    # Google requires a modern browser User-Agent header to return full HTML
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            print(
                f"[Google Scraper Warning]: Received HTTP {response.status_code}"
            )
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        # Google search result cards are usually enclosed in 'div.g' containers
        search_blocks = soup.select("div.g")

        for block in search_blocks:
            title_elem = block.select_one("h3")
            link_elem = block.select_one("a")
            # Snippets are usually stored in container classes like '.VwiC3b' or '[style*="-webkit-line-clamp"]'
            snippet_elem = block.select_one(
                "div.VwiC3b, div.IsZvec, div.yD8vcf"
            )

            if title_elem and link_elem and link_elem.get("href"):
                title = title_elem.get_text(strip=True)
                href = link_elem["href"]
                snippet = (
                    snippet_elem.get_text(strip=True) if snippet_elem else ""
                )

                # Skip non-http links or internal anchor tags
                if href.startswith("http"):
                    results.append(
                        {"title": title, "url": href, "snippet": snippet}
                    )

            if len(results) >= max_results:
                break

        if not results:
            return ""

        search_summary = "\n--- Live Google Search Results (Scraped) ---\n"
        for idx, item in enumerate(results, 1):
            search_summary += f"[{idx}] {item['title']}\nSnippet: {item['snippet']}\nURL: {item['url']}\n\n"
        search_summary += "--- End of Web Search Results ---\n"

        return search_summary

    except Exception as e:
        print(f"[Google Scraper Error]: {e}")
        return ""
    
# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Returns system status and VRAM usage."""
    vram_free_gb, vram_total_gb = 0.0, 0.0
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vram_free_gb = round(free / (1024**3), 2)
        vram_total_gb = round(total / (1024**3), 2)
    return {
        "status": "online",
        "active_model": manager.active_model_id or "None (Idle)",
        "vram_free_gb": vram_free_gb,
        "vram_total_gb": vram_total_gb,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

@app.get("/api/models")
async def get_models():
    """Returns the list of available models for the frontend dropdown."""
    return {"models": AVAILABLE_MODELS}

@app.get("/api/sessions")
async def list_sessions():
    """Returns a list of saved chat sessions."""

    os.makedirs(SESSIONS_DIR, exist_ok=True)

    files = [
        os.path.join(SESSIONS_DIR, f)
        for f in os.listdir(SESSIONS_DIR)
        if f.endswith(".json")
    ]

    files.sort(
        key=lambda x: os.path.getmtime(x),
        reverse=True
    )

    sessions = [
        os.path.splitext(os.path.basename(f))[0]
        for f in files
    ]

    return {
        "sessions": sessions
    }


@app.post("/api/load_session")
def load_session(req: LoadRequest):
    """Loads a specific chat history by ID."""

    if not req.session_id or req.session_id == "No Saved Chats":
        return {"history": []}

    session_id = os.path.basename(req.session_id)

    if not session_id.endswith(".json"):
        session_id += ".json"

    path = os.path.join(
        SESSIONS_DIR,
        session_id
    )

    if not os.path.isfile(path):
        return {"history": []}

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        return {
            "history": data.get("messages", [])
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/create_session")
def new_session():
    """Generates a new session ID based on timestamp."""

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    return {
        "session_id": f"chat_{timestamp}"
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deletes a saved chat session."""

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID is required"
        )

    # Prevent path traversal
    session_id = os.path.basename(session_id)

    if not session_id.endswith(".json"):
        session_id += ".json"

    path = os.path.join(
        SESSIONS_DIR,
        session_id
    )

    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    try:
        os.remove(path)

        return {
            "success": True,
            "session_id": os.path.splitext(session_id)[0]
        }

    except OSError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )

@app.post("/api/clear_vram")
async def manual_vram_clear():
    """Manually unloads the current model from the GPU."""
    async with manager.lock:
        manager.unload_vram()
    return {"status": "success", "message": "VRAM cleared", "new_session_id": f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"}

# ==============================================================================
# MAIN STREAMING ROUTE
# ==============================================================================
@app.post("/api/stream")
async def unified_stream_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str = Form(""),
    model: str = Form("qwen"),
    mode: str = Form("flash"),
    session_id: Optional[str] = Form(None),
    messages: Optional[str] = Form(None),
    max_tokens: int = Form(N_CTX),
    temperature: float = Form(0.7),
    web_search: bool = Form(True),
    file: Optional[UploadFile] = File(None)
):
    """
    Core generation endpoint. Handles file uploads, OCR, audio transcription, 
    web searches, context pruning, and streams Server-Sent Events (SSE) back to React.
    """
    background_tasks.add_task(cleanup_temp_uploads)
    mode2 = mode
    # 1. PARSE INCOMING MESSAGES
    parsed_messages = []
    if messages:
        try:
            parsed_messages = json.loads(messages)
        except Exception:
            parsed_messages = []

    user_prompt = prompt.strip()
    if not user_prompt and parsed_messages:
        user_prompt = extract_text_from_content(parsed_messages[-1].get("content", ""))

    # 2. FILE & MULTIMEDIA PROCESSING
    temp_path = None
    media_kind = "text"
    b64_file_contents = []

    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"media_{timestamp}_{uuid.uuid4().hex[:6]}{suffix}"
        temp_path = os.path.join(TEMP_UPLOADS_DIR, unique_filename)

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        mime_type, _ = mimetypes.guess_type(temp_path)
        mime_type = mime_type or "application/octet-stream"

        if mime_type.startswith("audio/"):
            import whisper
            whisper_model = whisper.load_model("base")
            result = await asyncio.to_thread(
                whisper_model.transcribe,
                temp_path,
                condition_on_previous_text=False,
                no_speech_threshold=0.6
            )
            transcription = result.get("text", "").strip() or "[Non-speech audio detected]"
            fallback_prompt = "Summarize or describe this audio clip."
            active_prompt = user_prompt if user_prompt else fallback_prompt
            user_prompt = f"Audio Transcription:\n\"{transcription}\"\n\nInstruction: {active_prompt}"
            media_kind = "audio"

        elif mime_type.startswith("video/"):
            fallback_prompt = "Describe what happens in this video in detail."
            if not user_prompt:
                user_prompt = fallback_prompt
            
            frames = extract_video_frames(temp_path, max_frames=8)
            if not frames:
                # if os.path.exists(temp_path):
                    #os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Could not extract frames from video file.")

            for frame in frames:
                buffered = io.BytesIO()
                frame.save(buffered, format="JPEG")
                b64_frame = base64.b64encode(buffered.getvalue()).decode("utf-8")
                b64_file_contents.append(f"data:image/jpeg;base64,{b64_frame}")
            media_kind = "video"

        elif mime_type.startswith("image/"):
            sanitize_image(temp_path)
            fallback_prompt = "Describe this image in detail or extract all text present."
            if not user_prompt:
                user_prompt = fallback_prompt

            with open(temp_path, "rb") as img_f:
                b64_image = base64.b64encode(img_f.read()).decode("utf-8")
            b64_file_contents.append(f"data:image/jpeg;base64,{b64_image}")
            media_kind = "image"

        else:
            processed = await process_uploaded_file(temp_path)
            if processed.get("text"):
                user_prompt += f"\n\nFILE CONTENT:\n{processed['text']}"
            media_kind = "document"

    # 3. WEB SEARCH INJECTION
    search_keywords = ["search", "latest", "news", "today", "who is", "what is the current"]
    should_search = web_search or any(kw in user_prompt.lower() for kw in search_keywords)
    
    if should_search and user_prompt:
        search_data = perform_google_scrape(user_prompt)
        if search_data:
            user_prompt = f"{search_data}\n\nUser Question: {user_prompt}"

    # 4. LOAD MODEL & CONFIG
    model_id = (model or "qwen").lower().strip()
    loaded_model, tokenizer_or_processor, config = await manager.load_model_by_config(model_id)
    is_chameleon = config.get("model_type") == "chameleon"
    backend_type = config.get("backend_type")

    # 5. SPECIAL NON-CHAT BACKENDS (MUSICGEN / IMAGE GEN)
    if backend_type == "musicgen":
        async def event_generator_music():
            start_time = time.time()
            os.makedirs("static/outputs", exist_ok=True)
            output_filename = f"music_{int(time.time())}.wav"
            output_path = os.path.join("static/outputs", output_filename)

            input_audio, sampling_rate = None, None
            if temp_path and media_kind == "audio":
                input_audio, sampling_rate = torchaudio.load(temp_path)

            if input_audio is not None:
                inputs = tokenizer_or_processor(text=[user_prompt], audio=input_audio, sampling_rate=sampling_rate, padding=True, return_tensors="pt").to(loaded_model.device)
            else:
                inputs = tokenizer_or_processor(text=[user_prompt], padding=True, return_tensors="pt").to(loaded_model.device)

            import numpy as np
            def generate_wav():
                gen_tokens = min(max_tokens, 750) if max_tokens else 256 
                with torch.no_grad():
                    audio_outputs = loaded_model.generate(**inputs, do_sample=True, guidance_scale=3.0, max_new_tokens=gen_tokens)
                audio_data = audio_outputs[0].cpu().float().numpy()
                if audio_data.ndim == 2:
                    audio_data = audio_data.T
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                target_sr = loaded_model.config.audio_encoder.sampling_rate
                scipy.io.wavfile.write(output_path, rate=int(target_sr), data=audio_int16)

            await asyncio.to_thread(generate_wav)

            audio_url = f"http://127.0.0.1:8000/static/outputs/{output_filename}"
            audio_response = f"\n\n🎵 **Generated Track:**\n<audio controls src=\"{audio_url}\"></audio>\n\n[Download Audio]({audio_url})\n\n"

            yield f"data: {json.dumps({'token': audio_response})}\n\n"
            # if temp_path and os.path.exists(temp_path):
                #os.remove(temp_path)

            elapsed = round(time.time() - start_time, 2)
            saved_id = append_and_save_chat(session_id=session_id, user_msg=user_prompt, assistant_msg=audio_response, model_used=config['name'], file_name=unique_filename if file else None)
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': 1, 'tps': 1}})}\n\n"

        return StreamingResponse(event_generator_music(), media_type="text/event-stream")

    is_image_generation_task = "generate image" in user_prompt.lower() or getattr(config, "is_diffusion", False)
    if is_image_generation_task and backend_type != "gguf":
        async def event_generator_image():
            start_time = time.time()
            os.makedirs("static/outputs", exist_ok=True)
            output_filename = f"gen_{int(time.time())}.png"
            output_path = os.path.join("static/outputs", output_filename)
            image_url = f"http://127.0.0.1:8000/static/outputs/{output_filename}"
            markdown_image = f"\n\n![Generated Image]({image_url})\n\n"

            yield f"data: {json.dumps({'token': markdown_image})}\n\n"
            # if temp_path and os.path.exists(temp_path):
                #os.remove(temp_path)

            elapsed = round(time.time() - start_time, 2)
            saved_id = append_and_save_chat(session_id=session_id, user_msg=user_prompt, assistant_msg=markdown_image, model_used=config['name'], file_name=unique_filename if file else None)
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': 1, 'tps': 1}})}\n\n"

        return StreamingResponse(event_generator_image(), media_type="text/event-stream")

    # 6. UNIFIED CHAT MESSAGE PARSING (FOR BOTH GGUF AND SAFETENSORS)
    unified_messages = []
    
    if parsed_messages:
        for msg in parsed_messages:
            unified_messages.append({
                "role": msg["role"],
                "content": extract_text_from_content(msg["content"])
            })

    # Ensure current user turn exists
    if not unified_messages or unified_messages[-1]["role"] != "user":
        unified_messages.append({"role": "user", "content": user_prompt})
    else:
        unified_messages[-1]["content"] = user_prompt

    # Prepend System Prompt
    if SYSTEM_PROMPT:
        if unified_messages[0]["role"] == "system":
            unified_messages[0]["content"] = SYSTEM_PROMPT
        else:
            unified_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # Apply Dynamic Context Pruning
    model_ctx_limit = N_CTX
    unified_messages = fit_messages_to_context(
        messages=unified_messages,
        tokenizer_or_model=tokenizer_or_processor if backend_type != "gguf" else loaded_model,
        max_context_limit=model_ctx_limit,
        max_generation_tokens=2048
    )

    # 7. ROUTE CHAT TO BACKEND
    if backend_type == "gguf":
        # Format multimodal payload for GGUF if images exist
        if b64_file_contents:
            for msg in reversed(unified_messages):
                if msg["role"] == "user":
                    text_content = msg["content"] if isinstance(msg["content"], str) else user_prompt
                    multimodal_content = [{"type": "text", "text": text_content}]
                    for img_b64 in b64_file_contents:
                        multimodal_content.append({
                            "type": "image_url",
                            "image_url": {"url": img_b64}
                        })
                    msg["content"] = multimodal_content
                    break

        async def event_generator_gguf():
            full_response, start_time, token_count = "", time.time(), 0
            try:
                def get_stream():
                    return loaded_model.create_chat_completion(
                        messages=unified_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9,
                        stream=True
                    )

                stream = await asyncio.to_thread(get_stream)
                for chunk in stream:
                    if await request.is_disconnected():
                        print("[SSE] Client disconnected mid-stream.")
                        break

                    token = ""
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            token = choice["delta"]["content"]
                        elif "text" in choice:
                            token = choice["text"]

                    if token:
                        full_response += token
                        token_count += 1
                        yield f"data: {json.dumps({'token': token})}\n\n"
                        await asyncio.sleep(0.001)

            except Exception as err:
                print(f"Error during GGUF stream iteration: {err}")
            finally:
                c=0
                # if temp_path and os.path.exists(temp_path):
                    #os.remove(temp_path)

            elapsed = round(time.time() - start_time, 2)
            tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
            saved_id = append_and_save_chat(
                session_id=session_id,
                user_msg=user_prompt,
                assistant_msg=full_response,
                model_used=f"{config['name']} ({media_kind.capitalize()})",
                file_name=unique_filename if file else None
            )
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': token_count, 'tps': tps}})}\n\n"

        return StreamingResponse(event_generator_gguf(), media_type="text/event-stream")

    else:
        # SAFETENSORS / TRANSFORMERS BACKEND
        processor = tokenizer_or_processor

        if is_chameleon:
            images = None
            chameleon_prompt = user_prompt
            if temp_path and media_kind == "image":
                with Image.open(temp_path) as img:
                    images = [img.convert("RGB")]
                if "<image>" not in chameleon_prompt:
                    chameleon_prompt = f"<image>\n{chameleon_prompt}"

            inputs = processor(text=chameleon_prompt, images=images, return_tensors="pt")
            target_dtype = getattr(loaded_model, "dtype", next(loaded_model.parameters()).dtype)
            inputs = {
                k: v.to(loaded_model.device, dtype=target_dtype) if torch.is_floating_point(v)
                else v.to(loaded_model.device)
                for k, v in inputs.items()
            }
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        else:
            tokenizer = processor
            formatted_input = tokenizer.apply_chat_template(unified_messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([formatted_input], return_tensors="pt").to(loaded_model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )

        thread = Thread(target=loaded_model.generate, kwargs=generation_kwargs)
        thread.start()

        async def event_generator_hf():
            full_response, start_time, token_count = "", time.time(), 0
            try:
                for new_text in streamer:
                    if await request.is_disconnected():
                        print("[SSE] Client disconnected mid-stream.")
                        break
                    full_response += new_text
                    token_count += 1
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
                    await asyncio.sleep(0.001)
            finally:
                c=0
                # if temp_path and os.path.exists(temp_path):
                    #os.remove(temp_path)

            elapsed = round(time.time() - start_time, 2)
            tps = round(token_count / elapsed, 1) if elapsed > 0 else 0
            saved_id = append_and_save_chat(
                session_id=session_id,
                user_msg=user_prompt,
                assistant_msg=full_response,
                model_used=f"{config['name']} ({media_kind.capitalize()})",
                file_name=unique_filename if unique_filename else None
            )
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': token_count, 'tps': tps}})}\n\n"

        return StreamingResponse(event_generator_hf(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)