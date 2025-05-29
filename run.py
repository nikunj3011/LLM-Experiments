import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())

# Set memory limit (10GB)
max_memory = 8 * 1024 * 1024 * 1024  # Bytes
def check_memory_usage():
    mem_info = psutil.virtual_memory()
    if mem_info.used >= max_memory:
        print("Memory usage exceeded limit! Consider optimizing.")


# Call this function in the loop or before loading the model
check_memory_usage()


torch.cuda.set_per_process_memory_fraction(0.8, 0)
# Set model path (local directory)
model_path = "Model"  # Update with the correct path

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("Model", load_in_4bit=True)
torch.cuda.empty_cache()

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

# Function to generate text
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Usage
text_output = generate_text("What is the meaning of life?")
print("Generated Response:", text_output)
