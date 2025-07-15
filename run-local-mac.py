import torch
from safetensors.torch import load_file
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


# Load weights
state_dict = load_file("custom_model.safetensors")

# Recreate model architecture
aa = TransformerLM(vocab_size=2112, embed_dim=256, num_heads=4, num_layers=4)
aa.load_state_dict(state_dict)
aa.eval()

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("custom_tokenizer.json")

def generate_text(model, tokenizer, prompt, max_len=100):
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(tokens).unsqueeze(0)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_ids)
            next_token = output[:, -1].argmax(dim=-1).item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# Example usage
prompt = "team adopted a new remote"
generated = generate_text(aa, tokenizer, prompt)
print(generated)
