from __future__ import annotations

import os
import clip
import torch
import faiss
import streamlit as st
from PIL import Image
import logging
import pickle
from typing import Dict, Any

# Configure logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # You can change to DEBUG for more verbosity
)
logger = logging.getLogger(__name__)
st.set_page_config(layout="wide")
# Set up device and load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Directory with pre-downloaded images
image_folder = "/Users/Z00F1YS/PycharmProjects/Target/hackathon/downloaded_images"
# image_folder = "/Users/Z00F1YS/Arjun/hackathon/2025/combined_images"
# image_folder = "/Users/Z00F1YS/PycharmProjects/Target/hackathon/targetdotcom_images"
# image_folder = "/Users/Z00F1YS/Arjun/hackathon/2025/final_dataset"
image_paths = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]

EMBEDDING_PATH = "cached_clip_embeddings.pkl"


def save_embeddings_to_disk(embeddings: torch.Tensor, image_paths: list[str]):
    data = {
        "embeddings": embeddings,
        "image_paths": image_paths
    }
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved embeddings and image paths to disk.")


def load_embeddings_from_disk() -> Dict[str, Any] | None:
    if os.path.exists(EMBEDDING_PATH):
        logger.info("Loading embeddings and image paths from disk...")
        with open(EMBEDDING_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource(show_spinner=True)
def get_image_embeddings_and_paths():
    data = load_embeddings_from_disk()
    if data is not None:
        return data["embeddings"], data["image_paths"]

    logger.info("No cached data found. Generating embeddings...")
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
            logger.info(f"[{idx + 1}/{len(image_paths)}] Encoded: {image_path}")
        except Exception as e:
            logger.error(f"Error with image {image_path}: {e}")

    embeddings_tensor = torch.cat(embeddings, dim=0)
    save_embeddings_to_disk(embeddings_tensor, valid_paths)
    return embeddings_tensor, valid_paths


# Get embeddings and paths
image_embeddings, image_paths = get_image_embeddings_and_paths()

# Build FAISS index once
faiss_index = faiss.IndexFlatIP(image_embeddings.shape[1])
faiss_index.add(image_embeddings.numpy())


# Define search function
def search_images(query: str, top_k: int = 10):
    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_embedding = model.encode_text(text)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    D, I = faiss_index.search(text_embedding.cpu().numpy(), top_k)
    return [image_paths[i] for i in I[0]]


def search_with_embedding(query_embedding: torch.Tensor, top_k: int = 10):
    D, I = faiss_index.search(query_embedding.cpu().numpy(), top_k)
    return [image_paths[i] for i in I[0]]


# Streamlit UI
# st.image("/Users/Z00F1YS/Downloads/target-logo.png", width=150)

col1, col2 = st.columns([1, 15])
with col1:
    st.image("/Users/Z00F1YS/Downloads/target-logo.png", width=100)
with col2:
    st.title("AssetIQ - Search made meaningful with embeddings")
# st.title("🔍 AssetIQ - Search made meaningful with embeddings")

# query = st.text_input("Enter a prompt to search for relevant images:", "")
#
# if query:
#     with st.spinner("Searching for relevant images..."):
#         results = search_images(query, top_k=12)
#
#     st.subheader(f"Top 10 Results for: '{query}'")
#     cols = st.columns(4)
#     for i, img_path in enumerate(results):
#         with cols[i % 4]:
#             st.image(img_path, caption=f"Result {i + 1}", use_column_width=True)

col1, col2 = st.columns(2)

with col1:
    text_query = st.text_input("Enter a text prompt")

with col2:
    uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "png", "jpeg"])

def display_images_grid(image_paths: list[str], columns: int = 5, width: int = 180):
    """
    Display images in a grid layout.

    Parameters:
    - image_paths: List of image file paths to display.
    - columns: Number of columns per row.
    - width: Width of each displayed image in pixels.
    """
    rows = (len(image_paths) + columns - 1) // columns  # Calculate number of rows needed
    for row_idx in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            img_idx = row_idx * columns + col_idx
            if img_idx < len(image_paths):
                with cols[col_idx]:
                    st.image(image_paths[img_idx],
                             caption=os.path.basename(image_paths[img_idx]),
                             width=width)


# Run search when either input is given
if text_query or uploaded_image:
    with torch.no_grad():
        if text_query:
            st.markdown(f"**Searching for text:** _{text_query}_")
            tokenized = clip.tokenize([text_query]).to(device)
            query_embedding = model.encode_text(tokenized)
        else:
            st.markdown(f"**Searching with uploaded image**")
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Query Image", width=250)
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            query_embedding = model.encode_image(image_tensor)

        # Normalize the embedding
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    # 🔍 Perform search
    results = search_with_embedding(query_embedding, top_k=10)
    st.markdown("### 🖼️ Top Matching Images")
    display_images_grid(results, columns=3, width=450)

else:
    st.info("Please enter a text prompt or upload an image.")
