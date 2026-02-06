import os
import zipfile
import cv2
import numpy as np
import streamlit as st
import tempfile
from PIL import Image

ZIP_PATH = "dataset.zip"
DATASET_DIR = "dataset"

if os.path.exists(ZIP_PATH) and not os.path.exists(DATASET_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    DATASET_DIR = "dataset"
    # Supported extensions
    VALID_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Tuning Parameters (Lower = Stricter)
    MAX_SHAPE_DIST = 10.0   # Max cv2.matchShapes distance to consider
    MAX_AR_DIFF = 0.5       # Max difference in Aspect Ratio allowed

# ==========================================
# ROBUST IMAGE PROCESSING
# ==========================================
def load_and_clean_image(image_source):
    """
    Reads image from path or bytes, cleans text/dimensions,
    and returns (Original, BinaryMask, LargestContour).
    """
    # 1. Load Image
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
    else:
        # Convert uploaded file (bytes) to OpenCV format
        file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

    # 2. Convert to Grayscale & Resize
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard width (speeds up processing, standardizes features)
    target_width = 800
    h, w = gray.shape
    scale = target_width / w
    gray = cv2.resize(gray, (target_width, int(h * scale)))

    # 3. Binarization (Invert if mostly white, assuming white paper)
    # Check average brightness to detect background color
    if np.mean(gray) > 127:
        # White background -> Invert
        thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    else:
        # Black background
        thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]

    # 4. Filter Noise (Text, Dimension Lines)
    # We assume the PART is the largest solid object in the drawing.
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return img, thresh, None

    # Sort contours by area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    largest_cnt = cnts[0]

    # Create a clean mask of JUST the largest contour
    clean_mask = np.zeros_like(thresh)
    cv2.drawContours(clean_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    return img, clean_mask, largest_cnt

def get_shape_properties(contour):
    """Calculates Aspect Ratio and Solidity for pre-filtering."""
    if contour is None: return 0, 0
    
    # Aspect Ratio (aligned rect)
    _, _, w, h = cv2.boundingRect(contour)
    ar = w / float(h) if h > 0 else 0
    
    # Solidity (Area / Convex Hull Area)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    return ar, solidity

# ==========================================
# DATABASE INDEXING
# ==========================================
@st.cache_data
def index_database(folder_path):
    """
    Scans the folder and caches contours in memory.
    Returns a list of dictionaries: [{'path': str, 'contour': obj, 'ar': float}, ...]
    """
    db_data = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Create if missing to avoid errors
        return []

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(Config.VALID_EXT)]
    
    for f in files:
        path = os.path.join(folder_path, f)
        try:
            _, _, contour = load_and_clean_image(path)
            if contour is not None and cv2.contourArea(contour) > 500:
                ar, solidity = get_shape_properties(contour)
                db_data.append({
                    'filename': f,
                    'path': path,
                    'contour': contour,
                    'ar': ar,
                    'solidity': solidity
                })
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return db_data

# ==========================================
# MAIN APP
# ==========================================
def main():
    st.set_page_config(page_title="Technical Drawing Matcher", layout="wide")
    st.title("🔩 Technical Drawing Shape Search")

    # 1. Sidebar Setup
    st.sidebar.header("Configuration")
    
    # Check dataset folder
    if not os.path.exists(Config.DATASET_DIR):
        st.error(f"⚠️ Folder '{Config.DATASET_DIR}' not found. Please create it and add images.")
        return

    # Load DB
    with st.spinner(f"Indexing '{Config.DATASET_DIR}'..."):
        db = index_database(Config.DATASET_DIR)
    
    st.sidebar.success(f"Indexed {len(db)} drawings.")
    if len(db) == 0:
        st.sidebar.warning("Folder is empty! Add .png/.jpg files.")

    # 2. File Upload
    uploaded_file = st.sidebar.file_uploader("Upload Query Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        # Reset file pointer for re-reading
        uploaded_file.seek(0)
        
        # Process Query
        q_img_orig, q_mask, q_cnt = load_and_clean_image(uploaded_file)
        
        if q_cnt is None:
            st.error("Could not detect a shape in the uploaded image. Try an image with higher contrast.")
        else:
            q_ar, q_solidity = get_shape_properties(q_cnt)
            
            # Display Query
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(q_mask, caption="Processed Shape Mask", use_container_width=True)
                st.caption(f"AR: {q_ar:.2f} | Sol: {q_solidity:.2f}")

            # 3. Matching Logic
            results = []
            
            # Progress bar for visual feedback
            progress_bar = st.progress(0)
            
            for i, item in enumerate(db):
                # Update progress
                progress_bar.progress((i + 1) / len(db))

                # --- STEP 1: Rough Filter (Aspect Ratio) ---
                # If the width/height ratio is completely different, skip it.
                ar_diff = abs(q_ar - item['ar'])
                if ar_diff > Config.MAX_AR_DIFF:
                    continue 

                # --- STEP 2: Precise Contour Match ---
                # matchShapes: Lower is better. 0 = Perfect match.
                # method=1 (I1) is usually best for technical shapes.
                shape_dist = cv2.matchShapes(q_cnt, item['contour'], cv2.CONTOURS_MATCH_I1, 0)
                
                # --- STEP 3: Scoring ---
                # We invert distance to get a score (0 to 100)
                # Matches > 1.0 distance are usually poor, but we keep them just in case.
                score = 100 / (1 + shape_dist * 5 + ar_diff * 2)

                results.append({
                    'path': item['path'],
                    'filename': item['filename'],
                    'score': score,
                    'raw_dist': shape_dist
                })

            progress_bar.empty()

            # Sort results
            results = sorted(results, key=lambda x: x['score'], reverse=True)

            # 4. Display Results
            with col2:
                st.subheader(f"Top Matches ({len(results)} found)")
                
                if not results:
                    st.warning("No shapes similar enough found.")
                
                # Dynamic Grid
                cols = st.columns(5)
                for idx, res in enumerate(results[:10]): # Show top 10
                    with cols[idx % 5]:
                        # Load image for display
                        display_img = Image.open(res['path'])
                        st.image(display_img, use_container_width=True)
                        
                        # Color code score
                        color = "green" if res['score'] > 70 else "orange" if res['score'] > 40 else "red"
                        st.markdown(f"**:{color}[Match: {int(res['score'])}%]**")
                        st.caption(f"Diff: {res['raw_dist']:.3f}")

if __name__ == "__main__":
    main()