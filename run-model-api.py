from datetime import datetime
import glob
import io
import os
import gc
import json
import asyncio
from pathlib import Path
import re
import base64
import mimetypes
from typing import Optional, List, Dict, Any
from threading import Thread
import uuid
import logging

from PIL import Image
import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import librosa
from pydantic import BaseModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer
)

# Imported for GGUF model support
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
from starlette.middleware.base import BaseHTTPMiddleware

# Logger setup
logger = logging.getLogger("uvicorn.error")

# ------------------------------------------------------------------------------
# APP SETUP & CORS
# ------------------------------------------------------------------------------
app = FastAPI(title="Dynamic VRAM Chat & Gemma-4 Vision API", version="3.0")
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

# Set max upload size to 50MB
app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=50 * 1024 * 1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration Constants
QWEN_MODEL_PATH = r"F:\models\code\Qwen2.5-Coder-7B-Instruct"
PHI_MOE_MODEL_PATH = Path(r"F:\models\moe\Qwen3-0.6B")
GEMMA_MODEL_PATH = r"D:\dev\LLM-Experiments\any-to-any\gemma"
QWEN35_GGUF_CLIP = r"D:\dev\LLM-Experiments\any-to-any\mmproj-F32.gguf"
QWEN35_GGUF_FILE = r"D:\dev\LLM-Experiments\any-to-any\qwen3.5-0.8b-Q4_K_M.gguf"

GEMMA_GGUF_FILE = r"D:\dev\LLM-Experiments\any-to-any\gemma\gemma-4-E4B-it-ultra-uncensored-heretic-Q8_0.gguf"
GEMMA_GGUF_MMPROJ = r"D:\dev\LLM-Experiments\any-to-any\gemma\gemma-4-E4B-it-mmproj-BF16.gguf"

SESSIONS_DIR = "./chat_history"
TEMP_UPLOADS_DIR = "./temp_uploads"
SYSTEM_PROMPT = "You are a pro in all fields especially in coding, a helpful AI assistant."

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# MODEL REGISTRY
# ------------------------------------------------------------------------------
AVAILABLE_MODELS = [
    {
        "id": "qwen",
        "name": "Qwen2.5 Coder 7B",
        "supports_vision": False,
        "description": "Code and general reasoning model (Default)"
    },
    {
        "id": "phi_moe",
        "name": "Microsoft Phi-tiny-MoE",
        "supports_vision": False,
        "description": "MOE and general reasoning model"
    },
    {
        "id": "gemma",
        "name": "Gemma-4 E4B Vision",
        "supports_vision": True,
        "description": "Multimodal vision, audio, and text model"
    },
    {
        "id": "qwen3.5-gguf",
        "name": "Qwen3.5 0.8B (GGUF)",
        "supports_vision": True,
        "description": "Fast lightweight GGUF model via llama-cpp"
    }
]

# ------------------------------------------------------------------------------
# DYNAMIC VRAM MODEL MANAGER
# ------------------------------------------------------------------------------
class DynamicModelManager:
    """
    Manages loading and unloading PyTorch & llama-cpp GGUF models from VRAM dynamically.
    Guarantees thread-safe transitions using asyncio.Lock.
    """
    def __init__(self):
        self.active_model_name: Optional[str] = None
        self.model = None
        self.tokenizer_or_processor = None
        self.lock = asyncio.Lock()

    def unload_vram(self):
        """Forces garbage collection and clears CUDA cache."""
        if self.model is not None:
            print(f"[VRAM Manager] Unloading '{self.active_model_name}' from VRAM...")
            del self.model
            del self.tokenizer_or_processor
            self.model = None
            self.tokenizer_or_processor = None
            self.active_model_name = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[VRAM Manager] VRAM cleared successfully.")

    async def load_qwen(self):
        """Loads Qwen2.5-Coder-7B in 4-bit mode."""
        async with self.lock:
            if self.active_model_name == "qwen":
                return self.model, self.tokenizer_or_processor

            print("[VRAM Manager] Requesting Qwen2.5-Coder. Swapping VRAM...")
            self.unload_vram()

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )

            print(f"[VRAM Manager] Loading Qwen tokenizer: {QWEN_MODEL_PATH}")
            self.tokenizer_or_processor = AutoTokenizer.from_pretrained(
                QWEN_MODEL_PATH,
                trust_remote_code=True,
                local_files_only=True
            )

            print(f"[VRAM Manager] Loading 4-bit Qwen model: {QWEN_MODEL_PATH}")
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            ).eval()

            self.active_model_name = "qwen"
            print("[VRAM Manager] Qwen2.5-Coder successfully loaded into VRAM.")
            return self.model, self.tokenizer_or_processor
        
    async def load_phi_moe(self):
        """Loads microsoft/Phi-tiny-MoE-instruct in 4-bit mode."""
        async with self.lock:
            if self.active_model_name == "phi_moe":
                return self.model, self.tokenizer_or_processor

            print("[VRAM Manager] Requesting Phi-tiny-MoE. Swapping VRAM...")
            self.unload_vram()

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )

            print(f"[VRAM Manager] Loading Phi-MoE tokenizer: {PHI_MOE_MODEL_PATH}")
            self.tokenizer_or_processor = AutoTokenizer.from_pretrained(
                PHI_MOE_MODEL_PATH,
                trust_remote_code=True,  # Critical for Phi MoE custom architecture
                local_files_only=True
            )

            print(f"[VRAM Manager] Loading 4-bit Phi-MoE model: {PHI_MOE_MODEL_PATH}")
            self.model = AutoModelForCausalLM.from_pretrained(
                PHI_MOE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,  # Critical for Phi MoE custom architecture
                local_files_only=True
            ).eval()

            self.active_model_name = "phi_moe"
            print("[VRAM Manager] Phi-tiny-MoE successfully loaded into VRAM.")
            return self.model, self.tokenizer_or_processor

    async def load_tiny_moe(self):
        """Loads Qwen2.5-Coder-7B in 4-bit mode."""
        async with self.lock:
            if self.active_model_name == "tiny_moe":
                return self.model, self.tokenizer_or_processor

            print("[VRAM Manager] Requesting tiny_moe. Swapping VRAM...")
            self.unload_vram()

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )

            print(f"[VRAM Manager] Loading Qwen tokenizer: {QWEN_MODEL_PATH}")
            self.tokenizer_or_processor = AutoTokenizer.from_pretrained(
                QWEN_MODEL_PATH,
                trust_remote_code=True,
                local_files_only=True
            )

            print(f"[VRAM Manager] Loading 4-bit Qwen model: {QWEN_MODEL_PATH}")
            self.model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            ).eval()

            self.active_model_name = "qwen"
            print("[VRAM Manager] Qwen2.5-Coder successfully loaded into VRAM.")
            return self.model, self.tokenizer_or_processor
            
    async def load_gemma_gguf(self):
        """Loads Gemma-4 E4B Q8_0 GGUF model using llama-cpp-python."""
        async with self.lock:
            if self.active_model_name == "gemma-gguf":
                return self.model, None

            print("[VRAM Manager] Requesting Gemma-4 GGUF. Swapping VRAM...")
            self.unload_vram()

            try:
                chat_handler = None

                # OPTIONAL: Enable multimodal chat handler if mmproj file exists
                if os.path.exists(GEMMA_GGUF_MMPROJ):
                    print(f"[VRAM Manager] Loading vision projector: {GEMMA_GGUF_MMPROJ}")
                    # Note: Use Llava15ChatHandler or the model's specific Handler
                    chat_handler = Qwen25VLChatHandler(clip_model_path=GEMMA_GGUF_MMPROJ)

                print(f"[VRAM Manager] Loading Gemma GGUF model: {GEMMA_GGUF_FILE}")
                self.model = Llama(
                    model_path=GEMMA_GGUF_FILE,
                    chat_handler=chat_handler,
                    chat_format="gemma" if chat_handler is None else None,
                    n_ctx=8192,         # Adjust context window as needed
                    n_gpu_layers=-1,    # Offload all layers to GPU VRAM
                    verbose=False
                )

                self.active_model_name = "gemma-gguf"
                print("[VRAM Manager] Gemma-4 GGUF successfully loaded into VRAM.")
                return self.model, None

            except Exception as e:
                logger.error(f"[VRAM Manager] Failed to load Gemma GGUF model: {e}", exc_info=True)
                self.model = None
                self.active_model_name = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
    async def load_gemma(self):
        """Loads Gemma-4-E4B-it multimodal model."""
        async with self.lock:
            if self.active_model_name == "gemma":
                return self.model, self.tokenizer_or_processor

            print("[VRAM Manager] Requesting Gemma-4. Swapping VRAM...")
            self.unload_vram()

            try:
                compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype
                )

                print(f"[VRAM Manager] Loading Gemma processor: {GEMMA_MODEL_PATH}")
                self.tokenizer_or_processor = AutoProcessor.from_pretrained(
                    GEMMA_MODEL_PATH,
                    trust_remote_code=True
                )

                print(f"[VRAM Manager] Loading Gemma-4 model: {GEMMA_MODEL_PATH}")
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    GEMMA_MODEL_PATH,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=compute_dtype
                ).eval()

                self.active_model_name = "gemma"
                print("[VRAM Manager] Gemma-4 successfully loaded into VRAM.")
                return self.model, self.tokenizer_or_processor

            except Exception as e:
                logger.error(f"[VRAM Manager] Failed to load Gemma-4 model: {e}", exc_info=True)
                self.model = None
                self.tokenizer_or_processor = None
                self.active_model_name = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

    async def load_qwen35_gguf(self):
        """Loads Qwen 3.5 GGUF Vision Model."""
        async with self.lock:
            if self.active_model_name == "qwen3.5-gguf":
                return self.model, None

            print("[VRAM Manager] Loading Qwen GGUF Vision Model...")
            self.unload_vram()

            try:
                chat_handler = Qwen25VLChatHandler(clip_model_path=QWEN35_GGUF_CLIP)

                self.model = Llama(
                    model_path=QWEN35_GGUF_FILE,
                    chat_handler=chat_handler,
                    n_ctx=8192,
                    n_gpu_layers=-1,
                    verbose=False
                )

                self.active_model_name = "qwen3.5-gguf"
                print("[VRAM Manager] Qwen Vision GGUF loaded successfully.")
                return self.model, None

            except Exception as e:
                logger.error("[VRAM Manager] Failed loading Qwen GGUF", exc_info=True)
                self.model = None
                self.active_model_name = None
                raise

    async def load_by_id(self, model_id: Optional[str] = "qwen"):
        """Generic dispatcher to load model by string ID (Defaults to Qwen)."""
        selected_id = (model_id or "qwen").lower().strip()
        
        if selected_id == "gemma":
            return await self.load_gemma_gguf()
        elif selected_id in ["qwen3.5-gguf", "qwen3.5_gguf", "gguf"]:
            return await self.load_qwen35_gguf()
        else:
            # Default fallback: Qwen
            return await self.load_qwen()


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

class SaveSessionRequest(BaseModel):
    session_id: str
    messages: List[ChatMessage]

class LoadRequest(BaseModel):
    session_id: str

class StreamRequestPayload(BaseModel):
    session_id: Optional[str] = None
    model: Optional[str] = "qwen"
    messages: List[ChatMessage]

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def clean_text_payload(text: Any) -> str:
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

def compile_all_user_history() -> Optional[str]:
    files = glob.glob(os.path.join(SESSIONS_DIR, "*.json"))
    files.sort(key=os.path.getmtime)
    all_user_inputs = []
    
    for fpath in files:
        filename = os.path.basename(fpath)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                msgs_list = data.get("messages", data.get("history", [])) if isinstance(data, dict) else data

                user_msgs = []
                for msg in msgs_list:
                    if isinstance(msg, dict) and msg.get("role") == "user":
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

def append_and_save_chat(
    session_id: Optional[str],
    user_msg: str,
    assistant_msg: str,
    model_used: str,
    file_preview: Optional[str] = None,
    file_name: Optional[str] = None
):
    """Appends new interaction to session JSON."""
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

    user_entry = {
        "role": "user",
        "content": user_msg
    }
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

def extract_video_frames(video_path: str, max_frames: int = 8) -> List[Image.Image]:
    """Extracts evenly spaced keyframes from a video file into PIL Images."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []

    step = max(1, total_frames // max_frames)
    frames = []
    
    for i in range(0, total_frames, step):
        if len(frames) >= max_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
    cap.release()
    return frames

# ------------------------------------------------------------------------------
# MODEL METADATA ROUTE FOR UI
# ------------------------------------------------------------------------------
@app.get("/api/models")
async def get_models():
    """Returns available models so UI can dynamically populate dropdowns."""
    return {"models": AVAILABLE_MODELS}

# ------------------------------------------------------------------------------
# SESSION MANAGEMENT & VRAM ROUTES
# ------------------------------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    try:
        files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
        sessions = [os.path.splitext(f)[0] for f in files]
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        return {"messages": []}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read session: {str(e)}")

@app.post("/api/sessions")
async def save_session(payload: Dict[str, Any]):
    session_id = payload.get("session_id")
    messages = payload.get("messages") or payload.get("history") or []

    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id in payload.")

    filename = session_id if session_id.endswith(".json") else f"{session_id}.json"
    filepath = os.path.join(SESSIONS_DIR, os.path.basename(filename))

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"messages": messages}, f, indent=2)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")

@app.post("/api/load_session")
def load_session(req: LoadRequest):
    if not req.session_id or req.session_id == "No Saved Chats":
        return {"history": []}

    filename = req.session_id if req.session_id.endswith(".json") else f"{req.session_id}.json"
    safe_filename = os.path.basename(filename)
    path = os.path.join(SESSIONS_DIR, safe_filename)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                history = data.get("history", data.get("messages", data)) if isinstance(data, dict) else data
                return {"history": history}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read session file: {str(e)}")

    return {"history": []}

@app.get("/api/create_session")
def new_session():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"session_id": f"chat_{timestamp}.json"}

@app.get("/api/summarize_prompt")
def get_summarize_prompt():
    prompt = compile_all_user_history()
    return {"prompt": prompt}

@app.post("/api/vram/clear")
@app.post("/api/clear_vram")
async def manual_vram_clear():
    async with manager.lock:
        manager.unload_vram()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return {"status": "success", "message": "VRAM cleared", "new_session_id": f"chat_{timestamp}.json"}

def sanitize_image(image_path: str) -> None:
    """
    Strips AI metadata (e.g., ComfyUI/A1111 workflows) and normalizes color space 
    to standard 8-bit RGB to prevent decoder crashes in llama-cpp / PIL.
    """
    try:
        with Image.open(image_path) as img:
            # Convert RGBA, Palette, or 16-bit channel modes cleanly to 8-bit RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Save fresh image, stripping all EXIF and metadata chunks
            img.save(image_path, format="JPEG", quality=95)
    except Exception as e:
        logger.error(f"Failed to sanitize image metadata: {e}")

# ------------------------------------------------------------------------------
# MULTIMODAL MEDIA INFERENCE ROUTE
# ------------------------------------------------------------------------------
@app.post("/api/chat")
async def process_chat(
    prompt: str = Form(""),
    model: str = Form("qwen"),
    session_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        has_file = file is not None and file.filename != ""

        if not has_file:
            raise HTTPException(
                status_code=400, 
                detail="No media attached. Use /api/stream for text-only messages."
            )

        suffix = os.path.splitext(file.filename)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"media_{timestamp}_{uuid.uuid4().hex[:6]}{suffix}"
        temp_path = os.path.join(TEMP_UPLOADS_DIR, unique_filename)

        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        try:
            mime_type, _ = mimetypes.guess_type(temp_path)
            mime_type = mime_type or "application/octet-stream"
            selected_model = (model or "qwen").lower().strip()
            user_text = prompt.strip()
            if mime_type.startswith("image/"):
                sanitize_image(temp_path)
                mime_type = "image/jpeg"
            # Helper for Audio transcription via Whisper -> GGUF
            async def run_audio_transcription_route(llm_model, model_name_tag):
                import whisper
                whisper_model = whisper.load_model("base")
                
                result = whisper_model.transcribe(
                    temp_path, 
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6
                )
                transcription = result.get("text", "").strip()

                if not transcription:
                    transcription = "[Non-speech audio detected: sound effects / ambient noise]"

                fallback_prompt = "Summarize or describe this audio clip."
                active_prompt = user_text if user_text else fallback_prompt

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Audio Transcription:\n\"{transcription}\"\n\nInstruction: {active_prompt}"
                    }
                ]

                response = llm_model.create_chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                    top_p=0.9
                )

                response_text = response["choices"][0]["message"]["content"]
                history_text = user_text if user_text else "[AUDIO ATTACHMENT]"

                saved_session_id = append_and_save_chat(
                    session_id=session_id,
                    user_msg=history_text,
                    assistant_msg=response_text.strip(),
                    model_used=f"{model_name_tag} (Audio)",
                    file_name=file.filename
                )

                return JSONResponse({
                    "session_id": saved_session_id,
                    "active_model": model_name_tag,
                    "media_type": "audio",
                    "response": response_text.strip()
                })

            # ------------------------------------------------------------------
            # 1. QWEN GGUF ROUTE
            # ------------------------------------------------------------------
            if selected_model in ["qwen3.5-gguf", "qwen3.5_gguf", "gguf"]:
                # Audio Fallback -> Route through Whisper STT + Gemma GGUF
                if mime_type.startswith("audio/"):
                    gemma_model, _ = await manager.load_gemma_gguf()
                    return await run_audio_transcription_route(gemma_model, "gemma-gguf (Audio Auto-Routed)")

                # Image & Video Handling via Qwen GGUF
                llm, _ = await manager.load_qwen35_gguf()
                message_content = []

                if mime_type.startswith("video/"):
                    fallback_prompt = "Describe what happens in this video in detail."
                    active_prompt = user_text if user_text else fallback_prompt
                    message_content.append({"type": "text", "text": active_prompt})

                    frames = extract_video_frames(temp_path, max_frames=8)
                    if not frames:
                        raise HTTPException(status_code=400, detail="Could not extract frames from video file.")

                    for frame in frames:
                        buffered = io.BytesIO()
                        frame.save(buffered, format="JPEG")
                        b64_frame = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}
                        })

                    media_kind = "video"
                    history_label = "[VIDEO ATTACHMENT]"

                else:
                    fallback_prompt = "Describe this image in detail or extract all text present."
                    active_prompt = user_text if user_text else fallback_prompt
                    message_content.append({"type": "text", "text": active_prompt})

                    with open(temp_path, "rb") as img_f:
                        b64_image = base64.b64encode(img_f.read()).decode("utf-8")
                    image_url = f"data:{mime_type};base64,{b64_image}"

                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })

                    media_kind = "image"
                    history_label = "[IMAGE ATTACHMENT]"

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message_content}
                ]

                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                    top_p=0.9
                )

                response_text = response["choices"][0]["message"]["content"]
                history_text = user_text if user_text else history_label

                saved_session_id = append_and_save_chat(
                    session_id=session_id,
                    user_msg=history_text,
                    assistant_msg=response_text.strip(),
                    model_used=f"Qwen3.5 GGUF ({media_kind.capitalize()})",
                    file_name=file.filename
                )

                return JSONResponse({
                    "session_id": saved_session_id,
                    "active_model": "qwen3.5-gguf",
                    "media_type": media_kind,
                    "response": response_text.strip()
                })

            # ------------------------------------------------------------------
            # 2. GEMMA MULTIMODAL ROUTE
            # ------------------------------------------------------------------
            else:
                gemma_model, _ = await manager.load_gemma_gguf()
                
                if mime_type.startswith("audio/"):
                    return await run_audio_transcription_route(gemma_model, "gemma-gguf")

                elif mime_type.startswith("video/"):
                    fallback_prompt = "Describe what happens in this video in detail."
                    active_prompt = user_text if user_text else fallback_prompt

                    frames = extract_video_frames(temp_path, max_frames=8)
                    if not frames:
                        raise HTTPException(status_code=400, detail="Could not extract frames from video file.")

                    message_content = [{"type": "text", "text": active_prompt}]
                    for frame in frames:
                        buffered = io.BytesIO()
                        frame.save(buffered, format="JPEG")
                        b64_frame = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}
                        })

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": message_content}
                    ]

                    response = gemma_model.create_chat_completion(
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.7,
                        top_p=0.9
                    )

                    response_text = response["choices"][0]["message"]["content"]
                    history_text = user_text if user_text else "[VIDEO ATTACHMENT]"
                    media_kind = "video"

                else:
                    fallback_prompt = "Describe this image in detail or extract all text present."
                    active_prompt = user_text if user_text else fallback_prompt

                    with open(temp_path, "rb") as img_f:
                        b64_image = base64.b64encode(img_f.read()).decode("utf-8")
                    image_url = f"data:{mime_type};base64,{b64_image}"

                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": active_prompt},
                                {"type": "image_url", "image_url": {"url": image_url}}
                            ]
                        }
                    ]

                    response = gemma_model.create_chat_completion(
                        messages=messages,
                        max_tokens=2048,
                        temperature=0.7,
                        top_p=0.9
                    )

                    response_text = response["choices"][0]["message"]["content"]
                    history_text = user_text if user_text else "[IMAGE ATTACHMENT]"
                    media_kind = "image"

                saved_session_id = append_and_save_chat(
                    session_id=session_id,
                    user_msg=history_text,
                    assistant_msg=response_text.strip(),
                    model_used=f"Gemma-4 E4B GGUF ({media_kind.capitalize()})",
                    file_name=file.filename
                )

                return JSONResponse({
                    "session_id": saved_session_id,
                    "active_model": "gemma-gguf",
                    "media_type": media_kind,
                    "response": response_text.strip()
                })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except HTTPException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ------------------------------------------------------------------------------
# UNIFIED SSE STREAMING ROUTE
# ------------------------------------------------------------------------------
@app.post("/api/stream")
async def stream_chat(payload: StreamRequestPayload):
    # Default to "qwen" if model field is not specified
    model_name = (payload.model or "qwen").lower().strip()
    session_id = payload.session_id

    # Get the latest user message from payload
    user_prompt = ""
    if payload.messages:
        user_prompt = extract_text_from_content(payload.messages[-1].content)

    # 1. GGUF STREAMING ROUTE
    if model_name in ["qwen3.5-gguf", "qwen3.5_gguf", "gguf"]:
        llm, _ = await manager.load_qwen35_gguf()
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in payload.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": extract_text_from_content(msg.content)
            })

        stream = llm.create_chat_completion(
            messages=formatted_messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )

        async def event_generator_gguf():
            full_response = ""
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    token = delta["content"]
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.001)
            
            saved_id = append_and_save_chat(session_id, user_prompt, full_response, "Qwen3.5-0.8B-GGUF")
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id})}\n\n"

        return StreamingResponse(event_generator_gguf(), media_type="text/event-stream")

    # 2. GEMMA MODEL STREAMING ROUTE
    elif model_name == 'gemma-4-vision':
        llm, _ = await manager.load_gemma_gguf()

        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in payload.messages:
            formatted_messages.append({
                "role": msg.role,
                "content": extract_text_from_content(msg.content)
            })

        stream = llm.create_chat_completion(
            messages=formatted_messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            stream=True
        )

        async def event_generator_gguf():
            full_response = ""
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    token = delta["content"]
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0.001)
            
            saved_id = append_and_save_chat(session_id, user_prompt, full_response, "Gemma-4-E4B-it")
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id})}\n\n"

        return StreamingResponse(event_generator_gguf(), media_type="text/event-stream")

    # 3. PYTORCH TRANSFORMERS STREAMING (QWEN DEFAULT ROUTE)
    elif model_name in ["phi", "phi_moe", "phi-tiny-moe"]:
        try:
            from transformers.utils.import_utils import is_torch_fx_available
        except ImportError:
            def is_torch_fx_available():
                return True
        loaded_model, tokenizer = await manager.load_phi_moe()

        messages = []
        for msg in payload.messages:
            text_content = extract_text_from_content(msg.content)
            messages.append({"role": msg.role, "content": text_content})

        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = tokenizer([formatted_input], return_tensors="pt").to(loaded_model.device)

        # Use tokenizer's eos_token_id or unk/pad fallback if pad_token_id is None
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_id
        )

        thread = Thread(target=loaded_model.generate, kwargs=generation_kwargs)
        thread.start()

        async def event_generator_hf():
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                yield f"data: {json.dumps({'token': new_text})}\n\n"
                await asyncio.sleep(0.01)
            
            saved_id = append_and_save_chat(
                session_id, 
                user_prompt, 
                full_response, 
                "Phi-tiny-MoE-instruct"
            )
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id})}\n\n"

        return StreamingResponse(event_generator_hf(), media_type="text/event-stream")
    else:
        loaded_model, tokenizer = await manager.load_qwen()

        messages = []
        for msg in payload.messages:
            text_content = extract_text_from_content(msg.content)
            messages.append({"role": msg.role, "content": text_content})

        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([formatted_input], return_tensors="pt").to(loaded_model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        thread = Thread(target=loaded_model.generate, kwargs=generation_kwargs)
        thread.start()

        async def event_generator_hf():
            full_response = ""
            for new_text in streamer:
                full_response += new_text
                yield f"data: {json.dumps({'token': new_text})}\n\n"
                await asyncio.sleep(0.01)
            
            saved_id = append_and_save_chat(session_id, user_prompt, full_response, "Qwen2.5-Coder-7B")
            yield f"data: {json.dumps({'done': True, 'session_id': saved_id})}\n\n"

        return StreamingResponse(event_generator_hf(), media_type="text/event-stream")

# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1",h11_max_incomplete_event_size=100 * 1024 * 1024, port=8000)