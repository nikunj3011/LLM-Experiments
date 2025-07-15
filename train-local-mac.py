#Prepare Your Dataset
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9.,!?\'\s]', '', text)  # Remove special characters
    return text.lower().strip()

with open("b.txt", "r") as f:
    raw_text = f.read()

cleaned_text = clean_text(raw_text)

#Tokenizer (Byte Pair Encoding from Scratch)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train_from_iterator([cleaned_text], trainer)
tokenizer.save("custom_tokenizer.json")

#Build Transformer Model (PyTorch)

import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embedding[:, :x.size(1)]
        x = self.transformer(x)
        return self.fc_out(x)


# Training the Model

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = [token_ids[i:i+seq_len] for i in range(len(token_ids)-seq_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1])
        y = torch.tensor(self.data[idx][1:])
        return x, y

# Tokenize your text
encoded = tokenizer.encode(cleaned_text).ids
dataset = TextDataset(encoded, seq_len=128)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TransformerLM(vocab_size=tokenizer.get_vocab_size(), embed_dim=256, num_heads=4, num_layers=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in loader:
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Save the model
import torch
from safetensors.torch import save_file

# Get model state dict
state_dict = model.state_dict()

# Save as .safetensors
save_file(state_dict, "custom_model.safetensors")


from safetensors.torch import load_file

# Load weights
loaded_weights = load_file("custom_model.safetensors")

# Load into model
model.load_state_dict(loaded_weights)

# Text Generation

def generate_text(model, tokenizer, prompt, max_len=50):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens).unsqueeze(0)
    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_ids)
            next_token = output[:, -1].argmax(dim=-1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

generate_text(model, tokenizer, "Once upon a time", max_len=50)

