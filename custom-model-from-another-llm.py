#pip install transformers datasets accelerate safetensors
from datasets import Dataset

# Load and clean your text
with open("b.txt", "r") as f:
    lines = f.readlines()

data = {"text": [line.strip() for line in lines if line.strip()]}
dataset = Dataset.from_dict(data)

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-neo-125M"  # Replace with a 256M model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # or tokenizer.unk_token

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()  # For causal language modeling
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./custom_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

from safetensors.torch import save_file

# Save model weights
save_file(model.state_dict(), "custom/custom_model_256M.safetensors")
save_model = AutoModelForCausalLM.from_pretrained("custom/custom_model_256M.safetensors")

# Save tokenizer
tokenizer.save_pretrained("custom/custom_model_tokenizer")
