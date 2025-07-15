import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import psutil

# Device setup for Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Memory check (8GB limit)
max_memory = 8 * 1024 * 1024 * 1024
def check_memory_usage():
    mem_info = psutil.virtual_memory()
    if mem_info.used >= max_memory:
        print("⚠️ Memory usage exceeded limit! Consider optimizing.")
check_memory_usage()

# Load Reddit-style summarization model
model_name = "MBZUAI/LaMini-Flan-T5-248M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
torch.mps.empty_cache()

# Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=0 if device == "cuda" else -1
)

# Sensory keyword dictionary
senses = {
    "sight": ["see", "look", "bright", "dark", "color", "glow"],
    "sound": ["hear", "loud", "quiet", "buzz", "echo", "whisper"],
    "smell": ["smell", "aroma", "stink", "fragrance", "scent"],
    "taste": ["taste", "sweet", "bitter", "sour", "salty"],
    "touch": ["feel", "soft", "rough", "smooth", "cold", "warm"]
}

def classify_senses(text):
    matched = {sense: [] for sense in senses}
    for sense, keywords in senses.items():
        for word in keywords:
            if word in text.lower():
                matched[sense].append(word)
    return {k: v for k, v in matched.items() if v}

# Format Reddit-style prompt
def format_prompt(post_title, post_body, comments):
    comment_block = "\n".join([f"- {c}" for c in comments])
    return f"""You're a witty Redditor. Here's a post and its top comments:

Title: {post_title}
Body: {post_body}

Top Comments:
{comment_block}

Respond in a tone that matches the Reddit community — insightful, funny, or sarcastic if needed."""

# Generate LLM response
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=300)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Reddit post
post_title = "Why do cats knock things off tables?"
post_body = "My cat keeps pushing stuff off my desk. Is it revenge or boredom?"
comments = [
    "They're just testing gravity. Still works.",
    "Cats are chaos incarnate.",
    "Mine does it while making eye contact. It's personal."
]

# Run analysis
all_text = " ".join([post_body] + comments)
emotions = emotion_classifier(all_text)
senses_detected = classify_senses(all_text)

# Generate response
prompt = format_prompt(post_title, post_body, comments)
response = generate_text(prompt)

# Output
print("🧠 Generated Reddit-style Response:\n", response)
print("\n🎭 Emotion Scores:\n", emotions)
print("\n👁️ Sensory Tags:\n", senses_detected)
