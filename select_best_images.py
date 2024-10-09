import sys
import os
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch.nn as nn
import numpy as np
import heapq
import shutil
import json
from safetensors import safe_open
from sklearn.cluster import KMeans

# Define the model and head loading function
def load_models(device):
    # Load the CLIP model from Hugging Face
    model_name_or_path = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name_or_path)
    processor = CLIPProcessor.from_pretrained(model_name_or_path)
    clip_model = clip_model.to(device)

    # Load weights from open_clip_pytorch_model.safetensors to CPU first
    weights_path = "open_clip_pytorch_model.safetensors"
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    clip_model.load_state_dict(state_dict, strict=False)

    # Move the model to the specified device
    clip_model = clip_model.to(device)

    # Define a linear layer to predict aesthetic score (adjust input dimension to 768)
    aesthetic_head = nn.Linear(768, 1)  # Removed ReLU activation
    aesthetic_head = aesthetic_head.to(device)
    aesthetic_head.eval()

    return clip_model, aesthetic_head, processor

# Function to calculate aesthetic score
def calculate_aesthetic_score(img_path, clip_model, aesthetic_head, processor, device):
    try:
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            # Get image features using CLIP model
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

            # Get the aesthetic score using the linear head
            score = aesthetic_head(image_features).item()

        return score, image_features.cpu().numpy().flatten()

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, None

# Function to calculate similarity score
def calculate_similarity_score(img_path, clip_model, processor, device, prompt="best event photo"):
    try:
        # Load and preprocess the image
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Encode text prompt
        text_inputs = processor(text=[prompt], return_tensors="pt").to(device)

        with torch.no_grad():
            # Get image and text features using CLIP model
            image_features = clip_model.get_image_features(**inputs)
            text_features = clip_model.get_text_features(**text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity score
            similarity_score = (image_features @ text_features.T).item()

        return similarity_score

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# Function to determine image orientation
def get_image_orientation(img_path):
    with Image.open(img_path) as img:
        width, height = img.size
        if width > height:
            return "landscape"
        elif width < height:
            return "portrait"
        else:
            return "square"

# Main function to handle scoring of multiple images
def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <group_size> <number_of_images> <path_to_images>")
        sys.exit(1)

    try:
        # Parse arguments
        group_size = int(sys.argv[1])
        num_images = int(sys.argv[2])
        image_dir = sys.argv[3]

        if group_size <= 0 or num_images <= 0 or not os.path.isdir(image_dir):
            print("Invalid group size, number of images, or path to images.")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, aesthetic_head, processor = load_models(device)

        # Load existing ratings from rating.json if it exists
        rating_file_path = os.path.join(image_dir, 'rating.json')
        if os.path.exists(rating_file_path):
            with open(rating_file_path, 'r') as f:
                image_ratings = json.load(f)
        else:
            image_ratings = {}

        # Process images
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(supported_formats)]
        total_images = len(image_files)
        image_scores = []
        image_features_list = []

        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_dir, img_file)
            if img_file in image_ratings:
                # Use cached score
                score = image_ratings[img_file]
                print(f"[{idx}/{total_images}] Using cached score for {img_file}: Score = {score:.4f}")
            else:
                # Calculate new score
                score, features = calculate_aesthetic_score(img_path, clip_model, aesthetic_head, processor, device)
                if score is not None:
                    image_ratings[img_file] = score
                    image_features_list.append((features, img_path))
                    print(f"[{idx}/{total_images}] Processed {img_file}: Score = {score:.4f}")

            # Track top N images
            if len(image_scores) < num_images:
                heapq.heappush(image_scores, (score, img_path))
            else:
                heapq.heappushpop(image_scores, (score, img_path))

        # Clustering to ensure diversity
        if len(image_features_list) > num_images:
            features = np.array([f[0] for f in image_features_list])
            kmeans = KMeans(n_clusters=num_images, random_state=0).fit(features)
            selected_indices = []
            for cluster_idx in range(num_images):
                cluster_points = [i for i, label in enumerate(kmeans.labels_) if label == cluster_idx]
                if cluster_points:
                    selected_indices.append(cluster_points[0])
            best_images = [image_features_list[i][1] for i in selected_indices]
        else:
            best_images = [img_path for _, img_path in image_scores]

        # Separate images by orientation
        landscape_images = [img for img in best_images if get_image_orientation(img) == "landscape"]
        portrait_images = [img for img in best_images if get_image_orientation(img) == "portrait"]

        # Create groups and ensure orientation consistency
        best_dir = os.path.join(image_dir, 'best')
        os.makedirs(best_dir, exist_ok=True)

        def create_groups(images, group_size, best_dir, start_group_number):
            group_number = start_group_number
            current_group = []

            for img_path in images:
                current_group.append(img_path)

                # If the current group reaches the group size, save it and start a new group
                if len(current_group) == group_size:
                    group_dir = os.path.join(best_dir, str(group_number))
                    os.makedirs(group_dir, exist_ok=True)
                    for img in current_group:
                        shutil.copy(img, group_dir)
                    print(f"\nGroup {group_number} has been created with {len(current_group)} images.")
                    group_number += 1
                    current_group = []

            # Save any remaining images in the last group
            if current_group:
                group_dir = os.path.join(best_dir, str(group_number))
                os.makedirs(group_dir, exist_ok=True)
                for img in current_group:
                    shutil.copy(img, group_dir)
                print(f"\nGroup {group_number} has been created with {len(current_group)} images.")

            return group_number

        # Create groups for landscape and portrait images separately
        next_group_number = create_groups(landscape_images, group_size, best_dir, 1)
        create_groups(portrait_images, group_size, best_dir, next_group_number)

        # Save updated ratings to rating.json
        with open(rating_file_path, 'w') as f:
            json.dump(image_ratings, f, indent=4)

    except ValueError:
        print("Invalid group size or number of images.")
        sys.exit(1)

if __name__ == '__main__':
    main()