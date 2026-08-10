import os
import uuid

# ==========================================
# SERVER & CLIENT CONFIG
# ==========================================
COMFYUI_SERVER = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

# ==========================================
# DYNAMIC PATHS
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "workflow.json")
VIDEO_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "video_workflow.json")
PROMPTS_FILE = os.path.join(SCRIPT_DIR, "prompts.json")
VIDEO_TASKS_FILE = os.path.join(SCRIPT_DIR, "video_tasks.json")

# ==========================================
# SAMPLING & QUOTA CONFIGS
# ==========================================
RANDOM_IMAGES = True
IMAGES_PER_ITEM = 1
RANDOM_POSITIONS_ITEM = 2
RANDOM_MINIMAL_POSITIONS_ITEM = 2
RANDOM_TEASE_POSITIONS_ITEM = 2
RANDOM_CAMERAS = 2

# ==========================================
# IMAGE WORKFLOW NODE IDs
# ==========================================
CLIP_TEXT_NODE_ID = "26"        # Positive Prompt Text Node
KSAMPLER_NODE_ID = "5"          # KSampler Node 1
KSAMPLER_NODE_2_ID = "10"       # KSampler Node 2

# ==========================================
# VIDEO WORKFLOW NODE IDs
# ==========================================
VIDEO_IMAGE_LOAD_NODE_ID = "262"     # LoadImage Node ("inputs.image")
VIDEO_PROMPT_NODE_ID = "260:124"     # MiniMaxH3ImageToVideo Node ("inputs.prompt")
VIDEO_SEED_NODE_ID = "260:15"        # RandomNoise Node ("inputs.noise_seed")
COMFYUI_INPUT_DIR = r"D:\Comfy-Desktop\ComfyUI-Shared\input"
# ==========================================
# GLOBAL PROMPTS
# ==========================================
GIRL_PROMPT = ",  "
GLOBAL_PROMPT = "" 
GLOBAL_PROMPT_MEN = " , "
END_GLOBAL_PROMPT = (
    ", "
)
ADDITIONAL_FILE_PROMPT_TEXT = (
    ""
)

