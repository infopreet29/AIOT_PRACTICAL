# pip install diffusers transformers torch accelerate
# pip install torchvision
# pip install -U huggingface_hub


import os
import re
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_TOKEN")
)

prompt = input("Describe the image you want: ")

image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")

# ---- Build a safe filename from the prompt ----
def make_safe_filename(text, max_length=50):
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)   # remove special characters
    text = re.sub(r'\s+', '_', text)           # spaces -> underscores
    return text[:max_length]

base_name = make_safe_filename(prompt)
filename = f"{base_name}.png"

# ---- Avoid overwriting: auto-increment if name exists ----
counter = 1
while os.path.exists(filename):
    filename = f"{base_name}_{counter}.png"
    counter += 1

image.save(filename)
print(f"Image saved as {filename}")
