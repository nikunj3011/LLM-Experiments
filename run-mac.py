import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psutil

# Check available device (Mac typically uses "mps" or "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Set memory limit (8GB for Mac)
max_memory = 8 * 1024 * 1024 * 1024  # Bytes
def check_memory_usage():
    mem_info = psutil.virtual_memory()
    if mem_info.used >= max_memory:
        print("Memory usage exceeded limit! Consider optimizing.")

check_memory_usage()

# Set model path (Hugging Face Model)
# model_name = "MBZUAI/LaMini-Flan-T5-248M"
model_name = "custom_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to Mac-compatible device
model.to(device)
torch.mps.empty_cache()  # Clear cache for Apple Metal optimization

# Function to generate text
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Usage
text_output = generate_text("What is the meaning of life?")
print("Generated Response:", text_output)
