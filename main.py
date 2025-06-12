from __future__ import annotations

import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import os
import time
import base64
import mimetypes
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import torch
from PIL import Image
import numpy as np
import requests
import io
import clip
import faiss
import streamlit as st
import logging
import pickle
from typing import Dict, Any, List

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# Allow origins (set this based on where your frontend will run)

origins = [
    "http://localhost:8080",  # your frontend
    "http://127.0.0.1:8080",  # sometimes frontend uses this
    "*"  # <-- while testing, allows all origins (not recommended for prod)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO  # You can change to DEBUG for more verbosity
# )
# logger = logging.getLogger(__name__)
# st.set_page_config(layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

# Load ControlNet model on GPU
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    # torch_dtype=torch.float16
).to(device)

# Load Stable Diffusion pipeline on GPU
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    controlnet=controlnet,
    # torch_dtype=torch.float16
).to(device)

# Generate depth map (on CPU to save GPU memory)
fe = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cpu").eval()

image_folder = "/Users/Z00F1YS/PycharmProjects/Target/hackathon/targetdotcom_images"
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

EMBEDDING_PATH = "cached_clip_embeddings_for_target_dot_com.pkl"


def save_embeddings_to_disk(embeddings: torch.Tensor, image_paths: list[str]):
    data = {
        "embeddings": embeddings,
        "image_paths": image_paths
    }
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(data, f)
    # logger.info("Saved embeddings and image paths to disk.")


def load_embeddings_from_disk() -> Dict[str, Any] | None:
    if os.path.exists(EMBEDDING_PATH):
        # logger.info("Loading embeddings and image paths from disk...")
        with open(EMBEDDING_PATH, "rb") as f:
            return pickle.load(f)
    return None


# @st.cache_resource(show_spinner=True)
def get_image_embeddings_and_paths():
    data = load_embeddings_from_disk()
    if data is not None:
        return data["embeddings"], data["image_paths"]

    # logger.info("No cached data found. Generating embeddings...")
    embeddings = []
    valid_paths = []

    for idx, image_path in enumerate(image_paths):
        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu())
                valid_paths.append(image_path)
            # logger.info(f"[{idx + 1}/{len(image_paths)}] Encoded: {image_path}")
        except Exception as e:
            pass
            # logger.error(f"Error with image {image_path}: {e}")

    embeddings_tensor = torch.cat(embeddings, dim=0)
    save_embeddings_to_disk(embeddings_tensor, valid_paths)
    return embeddings_tensor, valid_paths


# Get embeddings and paths
image_embeddings, image_paths = get_image_embeddings_and_paths()
id_to_image = {idx: path for idx, path in enumerate(image_paths)}

# Build FAISS index once
faiss_index = faiss.IndexFlatIP(image_embeddings.shape[1])
faiss_index.add(image_embeddings.numpy())


def convert_image_to_base64(file_path):
    """
    Converts an image file to base64 string.

    :param file_path: Path to the image file
    :return: Base64 encoded string of the image
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def get_think_tank_response(image_path):
    url = "https://thinktank.prod.target.com/v1/chat/completions"
    encoded_text = convert_image_to_base64(image_path)
    print("Image converted to encoding")

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate short descriptive keywords for the furnitures present in this image, avoid floor related details, try to include the specifics of the furniture like colour, material type, number of seatings"
                                "Also the output should be newline seperated, with each line mentioning only about one furniture at a time"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + encoded_text
                        }
                    }
                ]
            }
        ],
        "model": "gpt-4o",
        "stream": False,
        "max_tokens": 1024
    }
    headers = {
        "Authorization": "Bearer ******",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)
    return response.text


def read_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string


def encode_image_to_base64_with_data_uri(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"  # fallback

    return f"data:{mime_type};base64,{base64_str}"


def search_with_embedding(query_embedding: torch.Tensor, top_k: int = 10):
    D, I = faiss_index.search(query_embedding.cpu().numpy(), top_k)
    # return [image_paths[i] for i in I[0]]
    similar_images = []
    for idx, dist in zip(I[0], D[0]):
        image_path = id_to_image[idx]
        image_base64 = encode_image_to_base64_with_data_uri(image_path)
        similar_images.append(image_base64)

    return similar_images


# col1, col2 = st.columns([1, 17])
# with col1:
#     st.image("/Users/Z00F1YS/Downloads/target-logo.png", width=100)
# with col2:
#     st.title("AssetIQ - Search made meaningful with embeddings")
#
# col1, col2 = st.columns(2)
#
# with col1:
#     text_query = st.text_input("Enter a text prompt")
#
# with col2:
#     uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])
#
#
# def display_images_grid(image_paths: list[str], columns: int = 5, width: int = 180):
#     """
#     Display images in a grid layout.
#
#     Parameters:
#     - image_paths: List of image file paths to display.
#     - columns: Number of columns per row.
#     - width: Width of each displayed image in pixels.
#     """
#     rows = (len(image_paths) + columns - 1) // columns  # Calculate number of rows needed
#     for row_idx in range(rows):
#         cols = st.columns(columns)
#         for col_idx in range(columns):
#             img_idx = row_idx * columns + col_idx
#             if img_idx < len(image_paths):
#                 with cols[col_idx]:
#                     st.image(image_paths[img_idx],
#                              caption=os.path.basename(image_paths[img_idx]),
#                              width=width)
#
#
# # Run search when either input is given
# if text_query or uploaded_image:
#     with torch.no_grad():
#         if text_query:
#             st.markdown(f"**Searching for text:** _{text_query}_")
#             tokenized = clip.tokenize([text_query]).to(device)
#             query_embedding = model.encode_text(tokenized)
#         else:
#             st.markdown(f"**Searching with uploaded image**")
#             image = Image.open(uploaded_image).convert("RGB")
#             st.image(image, caption="Query Image", width=250)
#             image_tensor = preprocess(image).unsqueeze(0).to(device)
#             query_embedding = model.encode_image(image_tensor)
#
#         # Normalize the embedding
#         query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
#
#     # 🔍 Perform search
#     results = search_with_embedding(query_embedding, top_k=10)
#     st.markdown("### 🖼️ Top Matching Images")
#     display_images_grid(results)
#
# else:
#     st.info("Please enter a text prompt or upload an image.")


def decorate_room(image_bytes, prompt, furniture, output_path="decorated_room.png"):
    # Load and preprocess image from bytes
    room_image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((512, 512))
    inputs = fe(images=room_image, return_tensors="pt")
    furnitures = furniture.split(",")

    with torch.no_grad():
        depth = depth_model(**inputs).predicted_depth[0].numpy()

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_image = Image.fromarray(depth.astype(np.uint8)).resize((512, 512)).convert("RGB")

    pipe.safety_checker = None
    pipe.feature_extractor = None

    st = time.time()
    # Run image generation
    result = pipe(
        prompt=prompt,
        image=room_image,
        control_image=depth_image,
        strength=0.8,
        guidance_scale=9.0,
        num_inference_steps=30
    )

    output = result.images[0]
    print(f"Time taken to generate the image is {time.time() - st}s")
    output.save(output_path)
    response_text = get_think_tank_response(output_path)
    response_json = json.loads(response_text)
    keywords = response_json["choices"][0]["message"]["content"].split("\n")
    print(f"Keywords for the furnitures present in the image is {keywords}")
    # display(output)
    # return output

    # Collect FAISS search results
    search_results = {}

    # Run search when either input is given
    for keyword, fur in zip(keywords,furnitures):
        with torch.no_grad():
            tokenized = clip.tokenize([keyword]).to(device)
            query_embedding = model.encode_text(tokenized)

            # Normalize the embedding
            query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

        # 🔍 Perform search
        results = search_with_embedding(query_embedding, top_k=3)
        # Collect results per keyword
        search_results[fur] = results  # ensure 'results' is serializable

    return {
        "search_results": search_results,
        "ai_image": encode_image_to_base64_with_data_uri(output_path)
    }


@app.post("/recommend")
async def process_image(file: UploadFile = File(...), furniture: str = Query(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()

        print(f"Prompt to be fed is Furnish the room with a {furniture}, it should look realistic")

        # Call your internal processing function here
        # For example: output_text = your_image_processing_function(image_bytes)
        output_text = decorate_room(
            image_bytes,
            f"Furnish the room with a {furniture}, it should look realistic",
            furniture,
            "/Users/Z00F1YS/PycharmProjects/Target/hackathon/output/test_output.jpg"
        )

        # Return the output text
        return output_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
