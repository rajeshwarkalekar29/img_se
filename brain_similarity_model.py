import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import time
import requests
import json
import http.client
import urllib.parse
import re
import pickle

# ==============================================================================
# PART 1: BRAIN SIMILARITY ENGINE (Deep Learning Feature Extraction)
# ==============================================================================
class BrainSimilarityEngine:
    def __init__(self):
        # Use MobileNetV2 for a good balance between speed and accuracy
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            img_t = self.transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            with torch.no_grad():
                features = self.model(batch_t)
            return features.squeeze().numpy()
        except Exception:
            return None

# ==============================================================================
# PART 2: CLOUD INTELLIGENCE (OCR)
# ==============================================================================
class CloudIntelligence:
    def __init__(self, rapid_api_key):
        self.api_key = rapid_api_key
        self.host = "real-time-lens-data.p.rapidapi.com"

    def upload_temp_file(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                response = requests.post("https://file.io", files={"file": f})
            if response.status_code == 200:
                return response.json().get("link")
            return None
        except Exception:
            return None

    def get_image_text(self, image_url):
        conn = http.client.HTTPSConnection(self.host)
        headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.host
        }
        safe_url = urllib.parse.quote(image_url, safe='')
        try:
            conn.request("GET", f"/ocr?url={safe_url}&language=en", headers=headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            if data.get("status") == "OK":
                return data.get("data", {}).get("text", "").lower()
            return ""
        except Exception:
            return ""

def extract_die_number(text):
    match = re.search(r'die\s*no\.?\s*[:\-]?\s*(\d+)', text.lower())
    if match: return match.group(1)
    match = re.search(r'(\d{5})', text)
    if match: return match.group(1)
    return None

# ==============================================================================
# PART 3: MAIN EXECUTION
# ==============================================================================
def find_similar_images(query_path, dataset_folder, api_key, features_cache_path=None):
    print(f"\n--- BRAIN SIMILARITY PROCESSING: {os.path.basename(query_path)} ---")
    start_time = time.time()
    
    brain_engine = BrainSimilarityEngine()
    cloud_engine = CloudIntelligence(api_key)
    
    # STEP 1: OCR (Intelligence)
    api_text = ""
    public_url = cloud_engine.upload_temp_file(query_path)
    if public_url:
        api_text = cloud_engine.get_image_text(public_url)
    
    query_die_no = extract_die_number(api_text) if api_text else None
    if query_die_no:
        print(f"[INTELLIGENCE] Detected Die No: {query_die_no}")
    
    # STEP 2: Extract Query Features
    query_features = brain_engine.extract_features(query_path)
    if query_features is None:
        print("Error: Could not extract features from query image.")
        return

    # STEP 3: Scan Dataset
    print(f"[BRAIN] Scanning dataset for similarity...")
    results = []
    
    # Load pre-calculated features if available
    cached_features = {}
    if features_cache_path and os.path.exists(features_cache_path):
        with open(features_cache_path, 'rb') as f:
            cached_features = pickle.load(f)
        print(f"[BRAIN] Loaded {len(cached_features)} pre-calculated features.")

    if cached_features:
        # Fast path: use cached features
        for fname, feat in cached_features.items():
            # Cosine Similarity
            similarity = np.dot(query_features, feat) / (np.linalg.norm(query_features) * np.linalg.norm(feat))
            
            # Text Bonus
            text_bonus = 0
            if query_die_no and query_die_no in fname.lower():
                text_bonus = 0.5
            
            final_score = similarity + text_bonus
            results.append((final_score, fname, similarity))
    else:
        # Slow path: extract features on the fly
        search_files = glob.glob(os.path.join(dataset_folder, "*"))
        for f in search_files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.abspath(f) != os.path.abspath(query_path):
                feat = brain_engine.extract_features(f)
                if feat is not None:
                    similarity = np.dot(query_features, feat) / (np.linalg.norm(query_features) * np.linalg.norm(feat))
                    text_bonus = 0.5 if query_die_no and query_die_no in os.path.basename(f).lower() else 0
                    results.append((similarity + text_bonus, os.path.basename(f), similarity))

    # Sort by final score (higher is better)
    results.sort(key=lambda x: x[0], reverse=True)

    end_time = time.time()
    print(f"\n--- RESULTS (Processed in {end_time - start_time:.2f}s) ---")
    print(f"{'RANK':<5} | {'FILENAME':<25} | {'BRAIN SIMILARITY'}")
    print("-" * 50)
    
    for i, (score, fname, brain_sim) in enumerate(results[:10]):
        print(f"{i+1:<5} | {fname:<25} | {brain_sim:.4f}")

if __name__ == "__main__":
    MY_RAPID_KEY = "c19e1f2a59msh356118af08c17fap1e2c37jsn85f8cadc6f77"
    CACHE_PATH = "C:\\Users\\YASH\\Downloads\\product_images\\dataset_features.pkl"
    
    QUERY = input("Enter Query Image Path: ").strip().strip('"')
    DATASET = input("Enter Local Dataset Folder: ").strip().strip('"')
    
    if os.path.exists(QUERY) and os.path.exists(DATASET):
        find_similar_images(QUERY, DATASET, MY_RAPID_KEY, CACHE_PATH)
    else:
        print("Error: Invalid paths.")
