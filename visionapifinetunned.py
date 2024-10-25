import os
from openai import OpenAI
import requests
from PIL import Image
import io
import base64

def format(extension):
    if extension == "jpg":
        return "jpeg"
    elif extension == "png":
        return "png"
    elif extension == "gif":
        return "gif"
    elif extension == "webp":
        return "webp"
    return extension

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Load image
# img_path = "data/elephant.jpg"
# img_path = "data/grocery.jpg"
img_path = "data/alexis_ruhe01_1852_0018_022.tif"
extension = img_path.split(".")[-1]
img = Image.open(img_path)

# if image is not supported, convert to jpeg
if extension not in ["jpg", "jpeg", "png", "gif", "webp"] or img.mode != "RGB":
    img = img.convert("RGB")
    extension = "jpg"

# Convert image to base64

img_byte_arr = io.BytesIO()

img.save(img_byte_arr, format=format(extension).upper())

img_byte_arr = img_byte_arr.getvalue()

img_b64_str = base64.b64encode(img_byte_arr).decode()

img_type = "image/" + format(extension)

# Prompt
# prompt = "What is the name of the item in the image?"
prompt = "What does it say?"

# Send request
response = client.chat.completions.create(
    model="ft:gpt-4o-2024-08-06:personal:ocrd-testset:AM5ddcFv",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                },
            ],
        },
    ],
    stream=False,
)

response2 = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                },
            ],
        },
    ],
    stream=False,
)

print(response.choices[0].message)
print(response2.choices[0].message)

