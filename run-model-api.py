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

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    ChameleonForConditionalGeneration,
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
app.mount("/static", StaticFiles(directory="static"), name="static")

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
SYSTEM_PROMPT = "You are a pro in all fields especially in coding, a helpful AI assistant."

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# DYNAMIC MODEL REGISTRY
# ------------------------------------------------------------------------------
AVAILABLE_MODELS: List[Dict[str, Any]] = [
    {
        "id": "qwen",
        "name": "Qwen2.5 Coder 7B",
        "backend_type": "safetensors",
        "modality": "Text Generation",
        "supports_vision": False,
        "path": r"F:\models\code\Qwen2.5-Coder-7B-Instruct",
        "description": "Code and general reasoning model"
    },
    {
        "id": "phi_moe",
        "name": "Microsoft Phi-tiny-MoE",
        "backend_type": "safetensors",
        "modality": "Text Generation",
        "supports_vision": False,
        "path": r"F:\models\moe\Qwen3-0.6B",
        "description": "MOE reasoning model"
    },
    {
        "id": "gemma-gguf",
        "name": "Gemma-4 E4B (GGUF)",
        "backend_type": "gguf",
        "modality": "Any-to-Any",
        "supports_vision": True,
        "path": r"D:\dev\LLM-Experiments\any-to-any\gemma\gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf",
        "mmproj_path": r"D:\dev\LLM-Experiments\any-to-any\gemma\gemma-4-E4B-it-mmproj-BF16.gguf",
        "description": "Multimodal GGUF model via llama-cpp"
    },
    {
        "id": "qwen3.5-gguf",
        "name": "Qwen3.5 0.8B (GGUF)",
        "backend_type": "gguf",
        "modality": "Image-Text-to-Text",
        "supports_vision": True,
        "path": r"D:\dev\LLM-Experiments\any-to-any\qwen3.5-0.8b-Q4_K_M.gguf",
        "mmproj_path": r"D:\dev\LLM-Experiments\any-to-any\mmproj-F32.gguf",
        "description": "Fast lightweight GGUF vision model"
    },
    {
        "id": "chameleon-7b-plus",
        "name": "Chameleon-7b-plus",
        "backend_type": "safetensors",
        "modality": "Any-to-Any",
        "supports_vision": True,
        "model_type": "chameleon",
        "path": r"F:\models\any-to-any\chameleon-7b-hf",
        "description": "Meta Chameleon Any-to-Any safetensors"
    }
]

def get_model_config(model_id: str) -> Dict[str, Any]:
    selected_id = (model_id or "qwen").lower().strip()
    for m in AVAILABLE_MODELS:
        if m["id"].lower() == selected_id:
            return m
    return AVAILABLE_MODELS[0]

# ------------------------------------------------------------------------------
# DYNAMIC VRAM MODEL MANAGER
# ------------------------------------------------------------------------------
class DynamicModelManager:
    def __init__(self):
        self.active_model_id: Optional[str] = None
        self.active_config: Optional[Dict[str, Any]] = None
        self.model = None
        self.tokenizer_or_processor = None
        self.lock = asyncio.Lock()

    def unload_vram(self):
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
        config = get_model_config(model_id)
        target_id = config["id"]
        backend_type = config["backend_type"]

        async with self.lock:
            if self.active_model_id == target_id:
                return self.model, self.tokenizer_or_processor, config

            print(f"[VRAM Manager] Requesting '{target_id}' ({backend_type.upper()}). Swapping VRAM...")
            self.unload_vram()

            try:
                if backend_type == "gguf":
                    mmproj_path = config.get("mmproj_path")
                    valid_clip_path = mmproj_path if (mmproj_path and os.path.exists(mmproj_path)) else None

                    chat_handler = Qwen25VLChatHandler(clip_model_path=valid_clip_path)
                    if valid_clip_path:
                        print(f"[VRAM Manager] Binding Vision MMProj: {valid_clip_path}")
                    chat_format = "chatml" if "qwen" in target_id or "gemma" in target_id else None
                    print(f"[VRAM Manager] Loading GGUF Model: {config['path']}")
                    self.model = Llama(
                        model_path=config["path"],
                        clip_model_path=valid_clip_path,
                        chat_format=chat_format,
                        chat_handler=chat_handler,
                        n_gpu_layers=20,  
                        n_ctx=2048,
                        n_threads=6,
                        use_mmap=True,
                        use_mlock=False,
                        verbose=True
                    )
                    self.tokenizer_or_processor = None

                elif backend_type == "safetensors":
                    # Determine compute dtype dynamically
                    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype
                    )

                    model_path = config["path"]

                    if config.get("supports_vision"):
                        print(f"[VRAM Manager] Loading Vision Processor: {model_path}")
                        self.tokenizer_or_processor = AutoProcessor.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=os.path.exists(model_path)
                        )
                        
                        if config.get("model_type") == "chameleon":
                            print("[VRAM Manager] Loading Chameleon model")
                            self.model = ChameleonForConditionalGeneration.from_pretrained(
                                model_path,
                                quantization_config=bnb_config,
                                device_map="auto",
                                torch_dtype=compute_dtype,
                                trust_remote_code=True,
                                local_files_only=os.path.exists(model_path)
                            ).eval()

                            # REMOVE: self.model.dtype = compute_dtype
                            # DO THIS INSTEAD: Store on the manager or use a custom attribute name
                            self.compute_dtype = compute_dtype
                    else:
                        print(f"[VRAM Manager] Loading Tokenizer: {model_path}")
                        self.tokenizer_or_processor = AutoTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            local_files_only=os.path.exists(model_path)
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True,
                            local_files_only=os.path.exists(model_path)
                        ).eval()

                else:
                    raise ValueError(f"Unsupported backend type: {backend_type}")

                self.active_model_id = target_id
                self.active_config = config
                print(f"[VRAM Manager] Successfully loaded '{target_id}' into VRAM.")
                return self.model, self.tokenizer_or_processor, config

            except Exception as e:
                logger.error(f"[VRAM Manager] Failed to load model '{target_id}': {e}", exc_info=True)
                self.unload_vram()
                raise

manager = DynamicModelManager()

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
    for f in os.listdir(TEMP_UPLOADS_DIR):
        fpath = os.path.join(TEMP_UPLOADS_DIR, f)
        if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > 1800:
            try:
                os.remove(fpath)
            except Exception:
                pass

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

# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
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
    return {"models": AVAILABLE_MODELS}

@app.get("/api/sessions")
async def list_sessions():
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    sessions = [os.path.splitext(f)[0] for f in files]
    return {"sessions": sessions}

@app.post("/api/load_session")
def load_session(req: LoadRequest):
    if not req.session_id or req.session_id == "No Saved Chats":
        return {"history": []}

    filename = req.session_id if req.session_id.endswith(".json") else f"{req.session_id}.json"
    path = os.path.join(SESSIONS_DIR, os.path.basename(filename))

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"history": data.get("messages", data.get("history", []))}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"history": []}

@app.get("/api/create_session")
def new_session():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"session_id": f"chat_{timestamp}.json"}

@app.get("/api/summarize_prompt")
async def summarize_prompt():
    """Compiles all session histories into a summary prompt for the frontend."""
    all_content = []
    for file_name in os.listdir(SESSIONS_DIR):
        if file_name.endswith(".json"):
            file_path = os.path.join(SESSIONS_DIR, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    msgs = data.get("messages", data.get("history", []))
                    for m in msgs:
                        all_content.append(f"{m.get('role', 'user')}: {extract_text_from_content(m.get('content', ''))}")
            except Exception:
                pass
    if not all_content:
        return {"prompt": None}
    
    compiled_text = "\n".join(all_content[:200])  # Cap history token length
    return {"prompt": f"Please provide a concise summary of the following prior conversation topics:\n\n{compiled_text}"}

@app.post("/api/clear_vram")
async def manual_vram_clear():
    async with manager.lock:
        manager.unload_vram()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"status": "success", "message": "VRAM cleared", "new_session_id": f"chat_{timestamp}.json"}

# ------------------------------------------------------------------------------
# STREAMING ROUTE
# ------------------------------------------------------------------------------
@app.post("/api/stream")
async def unified_stream_chat(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str = Form(""),
    model: str = Form("qwen"),
    session_id: Optional[str] = Form(None),
    messages: Optional[str] = Form(None),
    max_tokens: int = Form(2048),
    temperature: float = Form(0.7),
    file: Optional[UploadFile] = File(None)
):
    background_tasks.add_task(cleanup_temp_uploads)

    model_id = (model or "qwen").lower().strip()
    loaded_model, tokenizer_or_processor, config = await manager.load_model_by_config(model_id)
    is_chameleon = config.get("model_type") == "chameleon"

    parsed_messages = []
    if messages:
        try:
            parsed_messages = json.loads(messages)
        except Exception:
            parsed_messages = []

    user_prompt = prompt.strip()
    if not user_prompt and parsed_messages:
        user_prompt = extract_text_from_content(parsed_messages[-1].get("content", ""))

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
                if os.path.exists(temp_path):
                    os.remove(temp_path)
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

    # --- 1. GGUF BACKEND ---
    # --- 1. GGUF BACKEND ---
    if config["backend_type"] == "gguf":
        formatted_messages = []

        # 1. Reconstruct chat history cleanly
        if parsed_messages:
            for msg in parsed_messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": extract_text_from_content(msg["content"])
                })
        
        # Ensure there is at least one message for the current user input
        if not formatted_messages or formatted_messages[-1]["role"] != "user":
            formatted_messages.append({"role": "user", "content": user_prompt})
        else:
            formatted_messages[-1]["content"] = user_prompt

        # 2. Inject System Prompt
        if SYSTEM_PROMPT:
            if formatted_messages[0]["role"] == "system":
                formatted_messages[0]["content"] = SYSTEM_PROMPT
            else:
                formatted_messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

        # 3. Format multimodal payload for the last user message
        if b64_file_contents:
            # Find the last 'user' role message to attach media
            for msg in reversed(formatted_messages):
                if msg["role"] == "user":
                    text_content = msg["content"] if isinstance(msg["content"], str) else user_prompt
                    multimodal_content = [{"type": "text", "text": text_content}]
                    
                    for img_b64 in b64_file_contents:
                        multimodal_content.append({
                            "type": "image_url",
                            "image_url": {"url": img_b64} # Ensures clean base64 string
                        })
                    
                    msg["content"] = multimodal_content
                    break

        async def event_generator_gguf():
            full_response = ""
            start_time = time.time()
            token_count = 0

            try:
                def get_stream():
                    # Multimodal GGUF MUST use create_chat_completion
                    return loaded_model.create_chat_completion(
                        messages=formatted_messages,
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
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            elapsed = round(time.time() - start_time, 2)
            tps = round(token_count / elapsed, 1) if elapsed > 0 else 0

            saved_id = append_and_save_chat(
                session_id=session_id,
                user_msg=user_prompt,
                assistant_msg=full_response,
                model_used=f"{config['name']} ({media_kind.capitalize()})",
                file_name=file.filename if file else None
            )

            yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': token_count, 'tps': tps}})}\n\n"

        return StreamingResponse(event_generator_gguf(), media_type="text/event-stream")

    # --- 2. SAFETENSORS / TRANSFORMERS BACKEND ---
    else:
        processor = tokenizer_or_processor
        is_image_generation_task = "generate image" in user_prompt.lower() or getattr(config, "is_diffusion", False)

        if is_image_generation_task:
            async def event_generator_image():
                start_time = time.time()
                os.makedirs("static/outputs", exist_ok=True)
                output_filename = f"gen_{int(time.time())}.png"
                output_path = os.path.join("static/outputs", output_filename)

                image_url = f"http://127.0.0.1:8000/static/outputs/{output_filename}"
                markdown_image = f"\n\n![Generated Image]({image_url})\n\n"

                yield f"data: {json.dumps({'token': markdown_image})}\n\n"

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

                elapsed = round(time.time() - start_time, 2)
                saved_id = append_and_save_chat(
                    session_id=session_id,
                    user_msg=user_prompt,
                    assistant_msg=markdown_image,
                    model_used=config['name'],
                    file_name=file.filename if file else None
                )
                yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': 1, 'tps': 1}})}\n\n"

            return StreamingResponse(event_generator_image(), media_type="text/event-stream")

        else:
            if is_chameleon:
                images = None
                chameleon_prompt = user_prompt

                if temp_path and media_kind == "image":
                    with Image.open(temp_path) as img:
                        images = [img.convert("RGB")]
                    
                    # Chameleon REQUIRES the `<image>` token in the prompt text!
                    if "<image>" not in chameleon_prompt:
                        chameleon_prompt = f"<image>\n{chameleon_prompt}"

                # Process text and image
                inputs = processor(text=chameleon_prompt, images=images, return_tensors="pt")

                # Cast float tensors (pixel_values) to match model precision
                target_dtype = getattr(loaded_model, "dtype", next(loaded_model.parameters()).dtype)
                inputs = {
                    k: v.to(loaded_model.device, dtype=target_dtype) if torch.is_floating_point(v) 
                    else v.to(loaded_model.device)
                    for k, v in inputs.items()
                }

                streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)    
            else:
                tokenizer = processor
                history = parsed_messages if parsed_messages else [{"role": "user", "content": user_prompt}]
                formatted_input = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                
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
                full_response = ""
                start_time = time.time()
                token_count = 0

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
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

                elapsed = round(time.time() - start_time, 2)
                tps = round(token_count / elapsed, 1) if elapsed > 0 else 0

                saved_id = append_and_save_chat(
                    session_id=session_id,
                    user_msg=user_prompt,
                    assistant_msg=full_response,
                    model_used=f"{config['name']} ({media_kind.capitalize()})",
                    file_name=file.filename if file else None
                )
                yield f"data: {json.dumps({'done': True, 'session_id': saved_id, 'metrics': {'elapsed_sec': elapsed, 'tokens': token_count, 'tps': tps}})}\n\n"

            return StreamingResponse(event_generator_hf(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)