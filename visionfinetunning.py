import os
import io
import base64
import requests
from PIL import Image
from openai import OpenAI
import zipfile
import json
import tempfile

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

# download zip
# Create a temporary directory
temp_dir = tempfile.TemporaryDirectory()

url = "https://github.com/tesseract-ocr/tesstrain/raw/main/ocrd-testset.zip"
zip_path = os.path.join(temp_dir.name, "ocrd-testset.zip")
extract_path = os.path.join(temp_dir.name, "ocrd-testset")

if not os.path.exists(zip_path):
    r = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(r.content)
else:
    print("File already exists")

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
else:
    print("Folder already exists")

# Load images (*.tif)
img_paths = [f for f in os.listdir(extract_path) if f.endswith(".tif")]

jsonl = []

# Load images
for img_path in img_paths:
    extension = img_path.split(".")[-1]
    img = Image.open(os.path.join(extract_path, img_path))

    # if image is not supported, convert to jpeg
    if extension not in ["jpg", "jpeg", "png", "gif", "webp"] or img.mode != "RGB":
        img = img.convert("RGB")
        extension = "jpg"

    # resize image by bicubic interpolation
    if img.size[0] > 512 or img.size[1] > 512:
        img.thumbnail((512, 512), Image.BICUBIC)

    # save image
    # if not os.path.exists("jpg"):
    #     os.makedirs("jpg")
    # temp_img_path = os.path.join("jpg", img_path.replace(".tif", ".jpg"))
    # img.save(temp_img_path)
    # print(f"Saved image to {temp_img_path}")

    # Convert image to base64
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format(extension).upper())
    img_byte_arr = img_byte_arr.getvalue()
    img_b64_str = base64.b64encode(img_byte_arr).decode()
    img_type = "image/" + format(extension)

    #label
    label_path = img_path.replace(".tif", ".gt.txt")
    with open(os.path.join(extract_path, label_path), "r", encoding="utf-8") as f:
        label = f.read().strip()

    # Prompt
    system_message = {"role": "system", "content": "You are an assistant that identifies text in images."}
    user_message = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64_str}", "detail": "low"}}
    ]}
    user_message2 = {"role": "user", "content": "What does it say?"}
    assistant_message = {"role": "assistant", "content": "The text in the image is: " + label}

    # append to jsonl
    data = {"messages": [system_message, user_message, user_message2, assistant_message]}
    jsonl.append(data)

# Save to jsonl
with open("results/ocrd-testset.jsonl", "w", encoding="utf-8") as f:
    for data in jsonl:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# and then use fine tuning UI at https://platform.openai.com/finetune
# drop off the jsonl file to the UI