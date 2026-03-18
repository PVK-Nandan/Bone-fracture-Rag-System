# ============================================================
# 🦴 Bone Fracture RAG System — Workflow & Architecture
# ============================================================
#
# Step 1 — Document Loading
# Purpose: Extract text from documents.
# Tools: pypdf, docx2txt
# Example output:
# { "source": "radiology_handbook.pdf", "page": 87, "text": "Hairline fractures..." }
#
# Step 2 — Text Chunking
# Tool: LangChain TextSplitter (chunk_size=500, overlap=100)
# Example: { "chunk_id": "chunk_1023", "text": "Hairline fractures..." }
#
# Step 3 — Metadata Extraction
# { "fracture_type": "hairline", "body_part": "wrist", "source": "...", "page": 87 }
#
# Step 4 — Store in Database (Weaviate)
# { "id": "chunk_1023", "embedding": [...], "text": "...", "metadata": {...} }
#
# ─────────────────────────────────────
# OFFLINE: Medical Docs → Extraction → Chunking → MedEmbed → Weaviate
# ONLINE:  X-ray → DenseNet169 → GradCAM → Query → Vector Search → Gemini → Diagnosis
# ─────────────────────────────────────
#
# Steps: 1.Doc loading 2.Chunking 3.Metadata 4.Embedding 5.DB storage
#        6.DenseNet169 7.GradCAM 8.Query gen 9.Vector search 10.Top-K
#        11.Gemini reasoning 12.Final diagnosis
# ============================================================


# %% ─────────────────────────────────────────────────────────────────────
# 📦 CELL 1 — Install ALL Dependencies (Run this, then RESTART RUNTIME)
# ─────────────────────────────────────────────────────────────────────────
import subprocess, sys

print('Cleaning up pre-installed conflicting packages...')
# Force uninstall numpy and pandas to avoid binary incompatibility
for p in ["numpy", "pandas"]:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", p], 
                   capture_output=True)

pkgs = [
    # Core data & ML (Install these first to pin stable versions)
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scikit-learn", 
    "scipy",
    
    # DL & Imaging
    "torch", "torchvision", "torchaudio",
    "monai", "timm", "torchmetrics", "seaborn", "tqdm",
    "Pillow", "opencv-python-headless", "grad-cam", "torchcam",

    # RAG & LLM packages
    "langchain", "langchain-community", "langchain-core", "langchain-text-splitters",
    "langchain-google-genai", "langchain-groq",
    "weaviate-client>=4.0.0",
    "sentence-transformers", "transformers", "huggingface-hub",
    "google-genai", "google-generativeai", "tenacity",

    # Document loaders
    "pypdf", "python-docx", "docx2txt", "pymupdf",
    "reportlab", "kaggle"
]

print('\nInstalling all required dependencies (this may take 2-3 minutes)...')
for p in pkgs:
    r = subprocess.run([sys.executable, "-m", "pip", "install", p, "-q"],
                       capture_output=True, text=True)
    print(f'  {"✅" if r.returncode==0 else "⚠️"} {p}')

print("\n" + "="*70)
print('🚨 CRITICAL: YOU MUST RESTART THE RUNTIME NOW! 🚨')
print('Go to: Runtime > Restart session (or Restart runtime)')
print('If you skip this step, Cell 2 will crash with a "numpy binary incompatibility" error.')
print("="*70)


# %% ─────────────────────────────────────────────────────────────────────
# 🔧 CELL 2 — Imports & Global Config
# ─────────────────────────────────────────────────────────────────────────
# 🛠️ SESSION SETTINGS
MODE = "DRIVE"  # Options: "DRIVE" (Persistent) or "MANUAL" (Temporary/Default)
GDRIVE_PATH = "Bone_Fracture_System" # Folder name in your Google Drive

import os, glob, json, time, base64, warnings, io, zipfile, math, re
from pathlib import Path
# --- BACKGROUND STATE PERSISTENCE ---
import pickle, threading

# Helper for timing experiment
TIME_TRACKER = {}
def record_time(step_name, start_time):
    TIME_TRACKER[step_name] = time.time() - start_time
    print(f"⏱️ {step_name} took: {TIME_TRACKER[step_name]:.2f} seconds")

class StateManager:
    def __init__(self, filename="checkpoint_state.pkl"):
        self.filename = filename
        self.lock = threading.Lock()
        self.data = {}
        
    @property
    def path(self):
        # Resolve path dynamically after GDRIVE_ROOT is set
        root = globals().get('GDRIVE_ROOT', '/tmp')
        return f"{root}/{self.filename}"

    def register(self, key, value):
        self.data[key] = value

    def save(self):
        if not globals().get('USE_DRIVE', False): return
        with self.lock:
            try:
                state = {k: globals().get(k) for k in self.data.keys() if k in globals()}
                with open(self.path, "wb") as f:
                    pickle.dump(state, f)
            except: pass

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    state = pickle.load(f)
                    for k, v in state.items():
                        globals()[k] = v
                print(f"📦 Global state restored from {self.path}")
            except:
                print("⚠️ Failed to restore state.")

    def auto_save_worker(self, interval=20):
        while True:
            self.save()
            time.sleep(interval)

# Initialize State Manager
STATE_MGR = StateManager()
# Note: loading is moved after path initialization below

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, cohen_kappa_score, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import cv2
warnings.filterwarnings('ignore')

# --- CONFIGURATION LOGIC ---
USE_DRIVE = False
if MODE.upper() == "DRIVE":
    from google.colab import drive
    try:
        print(f"⏳ Mounting Google Drive for persistence (Folder: {GDRIVE_PATH})...")
        drive.mount('/content/drive')
        GDRIVE_ROOT = f'/content/drive/MyDrive/{GDRIVE_PATH}'
        os.makedirs(GDRIVE_ROOT, exist_ok=True)
        
        DATA_ROOT    = f'{GDRIVE_ROOT}/data/MURA'
        OUTPUT_DIR   = f'{GDRIVE_ROOT}/outputs'
        DOC_DIR      = f'{GDRIVE_ROOT}/data/bone_docs'
        XRAY_DIR     = f'{GDRIVE_ROOT}/data/xrays'
        WEAVIATE_DIR = f'{GDRIVE_ROOT}/weaviate_persistence'
        KAG_DRIVE    = f'{GDRIVE_ROOT}/kaggle.json'
        USE_DRIVE = True
        print(f"✅ DRIVE MODE ACTIVE: Everything will be saved to {GDRIVE_ROOT}")
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}. Falling back to MANUAL mode.")
        MODE = "MANUAL"

if MODE.upper() == "MANUAL":
    print("🚀 MANUAL MODE ACTIVE: Using temporary local storage. (Data will be lost on session end)")
    GDRIVE_ROOT = None
    DATA_ROOT  = '/content/MURA'
    OUTPUT_DIR = '/content/mura_outputs'
    DOC_DIR    = '/content/bone_docs'
    XRAY_DIR   = '/content/xrays'
    WEAVIATE_DIR = '/content/weaviate_data'
    KAG_DRIVE    = None

# Initialize Directories
for d in [DATA_ROOT, OUTPUT_DIR, DOC_DIR, XRAY_DIR, WEAVIATE_DIR]:
    os.makedirs(d, exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# Finalize State Manager after paths are set
if MODE == "DRIVE":
    STATE_MGR.register("all_chunks", [])
    STATE_MGR.register("TIME_TRACKER", {})
    STATE_MGR.register("processed_files", [])
    STATE_MGR.load()
    threading.Thread(target=STATE_MGR.auto_save_worker, daemon=True).start()
    print("⏲️ Background Heartbeat: Saving state to Drive every 20s.")

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 320
BATCH_SIZE = 32
NUM_EPOCHS = 2
LR         = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE   = 10
SEED       = 42
THRESHOLD  = 0.5
FORCE_RETRAIN = False # Set to True if we want to ignore the saved model in Drive

print('━'*65)
print('  🦴 MURA Bone Fracture + RAG Integrated Pipeline (PERSISTENT)')
print('━'*65)
print(f'  Device   : {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type=="cuda" else ' ⚠️ Use GPU runtime!'))
print(f'  Persistent Dir : {GDRIVE_ROOT if USE_DRIVE else "LOCAL (TEMP)"}')
print(f'  Img Size : {IMG_SIZE}×{IMG_SIZE}')
print(f'  Batch    : {BATCH_SIZE}')
print(f'  Epochs   : {NUM_EPOCHS}')
print(f'  Seed     : {SEED}')
print('━'*65)


# %% ─────────────────────────────────────────────────────────────────────
# 🔑 CELL 3 — Gemini API Key
# ─────────────────────────────────────────────────────────────────────────
from google.colab import userdata, files as colab_files
from google import genai
from google.genai import types as gtypes

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    # Completely remove ALL invisible spaces and newlines (even in the middle of the key)
    GEMINI_API_KEY = "".join(GEMINI_API_KEY.split())
else:
    raise ValueError("❌ GEMINI_API_KEY not found in Colab Secrets.")

GEMINI_MODEL = "models/gemini-2.5-pro"
print(f"🔎 Using model: {GEMINI_MODEL}")

try:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options={"api_version": "v1"}
    )
    test = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents="Reply exactly with: GEMINI_OK"
    )
    USE_GEMINI = "GEMINI_OK" in test.text
    print(f"✅ Connected to {GEMINI_MODEL}")
except Exception as e:
    print(f"❌ Gemini error: {e}")
    USE_GEMINI = False
    gemini_client = None

print(f"\nGemini: {'ENABLED ✅' if USE_GEMINI else 'DISABLED ❌'}")

# Groq API key
try:
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    if GROQ_API_KEY: GROQ_API_KEY = GROQ_API_KEY.strip()
except Exception:
    GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY



# %% ─────────────────────────────────────────────────────────────────────
# 📥 CELL 4 — Dataset Retrieval + Timing Experiment
# ─────────────────────────────────────────────────────────────────────────
import os, subprocess, shutil

print('━'*60)
print('  📥 Dataset Retrieval Performance Check')
print('━'*60)

t_start = time.time()
MURA_EXISTS = os.path.exists(os.path.join(DATA_ROOT, 'MURA-v1.1'))

if MURA_EXISTS:
    print(f"✅ DRIVE HIT: MURA Dataset found in persistent storage.")
    record_time("Dataset Retrieval (Load)", t_start)
    TIME_TRACKER["Dataset_Status"] = "Loaded from Drive (Fast)"
else:
    print("⏳ DRIVE MISS: Starting fresh Kaggle download...")
    # Handle Kaggle Credentials
    KAG_CONFIG = os.path.expanduser('~/.kaggle')
    os.makedirs(KAG_CONFIG, exist_ok=True)
    if USE_DRIVE and os.path.exists(KAG_DRIVE):
        shutil.copy(KAG_DRIVE, os.path.join(KAG_CONFIG, 'kaggle.json'))
    else:
        up_kag = colab_files.upload()
        if 'kaggle.json' in up_kag:
            with open(os.path.join(KAG_CONFIG, 'kaggle.json'), "wb") as f:
                f.write(up_kag['kaggle.json'])
            if USE_DRIVE: shutil.copy(os.path.join(KAG_CONFIG, 'kaggle.json'), KAG_DRIVE)
    os.chmod(os.path.join(KAG_CONFIG, 'kaggle.json'), 0o600)

    # Download
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'cjinny/mura-v11', '-p', DATA_ROOT, '--unzip'])
    record_time("Dataset Retrieval (Download)", t_start)
    TIME_TRACKER["Dataset_Status"] = "Downloaded from Kaggle (Slow)"

# Stats
BODY_PARTS = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']
total_imgs = 0
mura_base = os.path.join(DATA_ROOT, 'MURA-v1.1')
for part in BODY_PARTS:
    train_path = os.path.join(mura_base, 'train', part)
    if os.path.exists(train_path):
        imgs = glob.glob(os.path.join(train_path, '**', '*.png'), recursive=True)
        print(f'  {part:<15}: {len(imgs):5d} images')
        total_imgs += len(imgs)
print(f'\n  Total images in storage: {total_imgs}')


# %% ─────────────────────────────────────────────────────────────────────
# 🗂️ CELL 5 — Load & Split MURA Images
# ─────────────────────────────────────────────────────────────────────────
def load_mura_samples(data_root):
    samples = []
    mura_root = os.path.join(data_root, 'MURA-v1.1')
    if not os.path.exists(mura_root):
        mura_root = data_root
    for split in ['train', 'valid']:
        split_path = os.path.join(mura_root, split)
        if not os.path.exists(split_path): continue
        for part in os.listdir(split_path):
            part_path = os.path.join(split_path, part)
            if not os.path.isdir(part_path): continue
            for patient in os.listdir(part_path):
                pat_path = os.path.join(part_path, patient)
                if not os.path.isdir(pat_path): continue
                for study in os.listdir(pat_path):
                    study_path = os.path.join(pat_path, study)
                    if not os.path.isdir(study_path): continue
                    label = 1 if 'positive' in study.lower() else 0
                    for img_file in os.listdir(study_path):
                        if img_file.lower().endswith(('.png','.jpg','.jpeg')):
                            samples.append({
                                'path': os.path.join(study_path, img_file),
                                'label': label,
                                'part': part.replace('XR_',''),
                                'patient': patient,
                                'split': split,
                            })
    return samples

all_samples = load_mura_samples(DATA_ROOT)
df_all = pd.DataFrame(all_samples)
print(f'Loaded {len(all_samples)} total MURA images')
print(f'  Normal: {(df_all["label"]==0).sum()}  |  Abnormal: {(df_all["label"]==1).sum()}')

train_df = df_all[df_all['split']=='train']
test_df_official = df_all[df_all['split']=='valid']

train_samples, val_samples = train_test_split(
    train_df.to_dict('records'), test_size=0.15,
    random_state=SEED, stratify=train_df['label']
)
test_samples = test_df_official.to_dict('records')

print(f'\nSplit: Train={len(train_samples)} | Val={len(val_samples)} | Test={len(test_samples)}')


# %% ─────────────────────────────────────────────────────────────────────
# 🖼️ CELL 6 — Dataset Class + DataLoaders
# ─────────────────────────────────────────────────────────────────────────
MURA_MEAN = [0.485, 0.456, 0.406]
MURA_STD  = [0.229, 0.224, 0.225]
BODY_PARTS_SHORT = ['WRIST','ELBOW','FINGER','SHOULDER','HUMERUS','FOREARM','HAND']

def get_transforms(train=True):
    if train:
        return T.Compose([
            T.Grayscale(3), T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(), T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(), T.Normalize(MURA_MEAN, MURA_STD),
        ])
    return T.Compose([
        T.Grayscale(3), T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(), T.Normalize(MURA_MEAN, MURA_STD),
    ])

class MURADataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = [s for s in samples if not os.path.basename(s['path']).startswith("._")]
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        try: img = Image.open(s['path']).convert('L')
        except: img = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))
        if self.transform: img = self.transform(img)
        return {'image': img, 'label': torch.tensor(s['label'], dtype=torch.float32),
                'part': s['part'], 'path': s['path']}

train_ds = MURADataset(train_samples, get_transforms(True))
val_ds   = MURADataset(val_samples,   get_transforms(False))
test_ds  = MURADataset(test_samples,  get_transforms(False))

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")


# %% ─────────────────────────────────────────────────────────────────────
# 🧠 CELL 7 — Train DenseNet169 (Persistent Check)
# ─────────────────────────────────────────────────────────────────────────
class MURAModel(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat_dim, 512), nn.ReLU(),
            nn.Dropout(dropout*0.5), nn.Linear(512, 1),
        )
    def forward(self, x):
        return self.classifier(self.encoder(x)).squeeze(1)

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        'auc': roc_auc_score(labels, probs),
        'kappa': cohen_kappa_score(labels, preds),
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
    }

model = MURAModel('densenet169', pretrained=True, dropout=0.3).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.2]).to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# %% ─────────────────────────────────────────────────────────────────────
# 🧠 CELL 7 — Train DenseNet169 + Persistence Experiment
# ─────────────────────────────────────────────────────────────────────────
t_start_model = time.time()
ckpt_path = os.path.join(OUTPUT_DIR, 'best_densenet169_mura.pth')
IF_MODEL_EXISTS = os.path.exists(ckpt_path)

if IF_MODEL_EXISTS and not FORCE_RETRAIN:
    print(f'🚀 DRIVE HIT: Loading pre-trained model from {ckpt_path}')
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['state'])
    record_time("Model Initialization (Load)", t_start_model)
    TIME_TRACKER["Model_Status"] = "Loaded from Drive (Fast)"
else:
    print('🔄 DRIVE MISS: Starting training phase...')
    best_kappa, best_state, patience_ctr = -1.0, None, 0
    t_train_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tr_losses = []
        for batch in train_loader:
            imgs, lbls = batch['image'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_probs, val_lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch['image'].to(DEVICE))
                val_probs.extend(torch.sigmoid(out).cpu().numpy())
                val_lbls.extend(batch['label'].numpy())

        vm = compute_metrics(np.array(val_lbls), np.array(val_probs))
        scheduler.step()

        tag = ''
        if vm['kappa'] > best_kappa:
            best_kappa = vm['kappa']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({'epoch': epoch, 'state': best_state, 'kappa': best_kappa, 'auc': vm['auc']}, ckpt_path)
            patience_ctr = 0; tag = ' ⭐'
        else:
            patience_ctr += 1

        print(f'  Ep {epoch:03d}/{NUM_EPOCHS} | AUC={vm["auc"]:.4f} | κ={vm["kappa"]:.4f}{tag}')
        if patience_ctr >= PATIENCE:
            print(f'\n⚠️ Early stopping at epoch {epoch}')
            break
    
    if best_state:
        model.load_state_dict(best_state)
    record_time("Model Initialization (Train)", t_start_model)
    TIME_TRACKER["Model_Status"] = "Trained from Scratch (Slow)"


# %% ─────────────────────────────────────────────────────────────────────
# 🩻 CELL 8 — GradCAM + Gemini Trauma Analysis Functions
# ─────────────────────────────────────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

def get_gradcam_heatmap(model, img_tensor, pil_img_gray, device):
    target_layers = [model.encoder.features.norm5]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0).to(device),
                        targets=[BinaryClassifierOutputTarget(1)])[0]

    orig_np = np.array(pil_img_gray.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    orig_np = orig_np / orig_np.max() if orig_np.max() > 0 else orig_np
    cam_resized = cv2.resize(grayscale_cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    xray_rgb = np.stack([orig_np] * 3, axis=-1)
    overlay = ((0.6) * xray_rgb + 0.4 * heatmap).clip(0, 1)
    return (overlay * 255).astype(np.uint8), cam_resized

TRAUMA_FEATURE_PROMPT = """You are a board-certified radiologist AI assistant. Analyse the bone X-ray image and return ONLY valid JSON (no markdown fences):

{
  "overall_abnormality": "normal" | "abnormal",
  "confidence": <float 0-1>,
  "body_part_detected": "<string>",
  "trauma_findings": {
    "bone_fracture": {
      "detected": true | false,
      "fracture_types": {
        "complete_fracture":        {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "incomplete_greenstick":    {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "hairline_fracture":        {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "comminuted_fracture":      {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "displaced_fracture":       {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "non_displaced_fracture":   {"detected": true|false, "confidence": <float>, "location": "<string>"},
        "stress_fracture":          {"detected": true|false, "confidence": <float>, "location": "<string>"}
      }
    },
    "dislocation": {
      "detected": true | false,
      "confidence": <float>,
      "joint_affected": "<string>",
      "direction": "<string>"
    }
  },
  "clinical_summary": "<one-paragraph summary>"
}"""

def gemini_trauma_analysis(image_path, client, model_name):
    try:
        img = Image.open(image_path).convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        response = client.models.generate_content(
            model=model_name,
            contents=[gtypes.Part.from_bytes(data=buf.getvalue(), mime_type='image/png'),
                      TRAUMA_FEATURE_PROMPT],
        )
        text = response.text.strip()
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"  ⚠️ Gemini failed: {e}")
        return None

print('✅ CELL 8 ready — GradCAM + Gemini functions loaded')


# %% ─────────────────────────────────────────────────────────────────────
# 🏆 CELL 9 — Full Test Set Evaluation + SOTA Leaderboard
# ─────────────────────────────────────────────────────────────────────────
print('📊 Evaluating DenseNet169 on full MURA test set...')
model.eval()
all_probs_test, all_labels_test = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing', leave=False):
        logits = model(batch['image'].to(DEVICE))
        all_probs_test.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels_test.extend(batch['label'].numpy())

all_probs_test  = np.array(all_probs_test)
all_labels_test = np.array(all_labels_test)
all_preds_test  = (all_probs_test >= THRESHOLD).astype(int)

our_auc   = roc_auc_score(all_labels_test, all_probs_test)
our_kappa = cohen_kappa_score(all_labels_test, all_preds_test)
our_acc   = accuracy_score(all_labels_test, all_preds_test)

print(f'\n✅ Our DenseNet169: AUC={our_auc:.4f} | κ={our_kappa:.4f} | Acc={our_acc:.4f}')

leaderboard = [
    ('Our DenseNet169 (This Work)',  our_auc, our_kappa, our_acc,  'Trained on MURA',      '#FFD700'),
    ('DenseNet169 (Stanford)',       0.929,   0.71,      0.82,     'Rajpurkar 2018',        '#FF6B6B'),
    ('Calibrated Ensemble',          0.930,   0.73,      0.84,     'Sci Reports 2021',      '#FF8C42'),
    ('EfficientNet-B4',              0.947,   0.78,      0.87,     'MURA fine-tuned 2022',  '#A855F7'),
    ('ResNet-152 (Shoulder)',        0.940,   0.76,      0.96,     'Shoulder 2021',         '#EC4899'),
    ('HyperCol-CBAM-DenseNet',       0.914,   0.68,      0.80,     'PMC Wrist 2023',        '#14B8A6'),
    ('SKELEX Foundation',            0.910,   0.65,      0.79,     'SKELEX 2024',           '#F97316'),
    ('YOLOv8 (Pediatric)',           0.890,   0.60,      0.82,     'GRAZPEDWRI-DX 2023',   '#06B6D4'),
]
leaderboard.sort(key=lambda x: x[1], reverse=True)

print('\n🏆 SOTA Leaderboard:')
for rank, e in enumerate(leaderboard, 1):
    medal = '🥇🥈🥉'[rank-1] if rank<=3 else f'#{rank}'
    print(f'  {medal} {e[0]:<30} AUC={e[1]:.3f} κ={e[2]:.3f} Acc={e[3]:.3f}')


# %% ─────────────────────────────────────────────────────────────────────
# 🕸️ CELL 10 — Radar Chart
# ─────────────────────────────────────────────────────────────────────────
our_sens = recall_score(all_labels_test, all_preds_test)
tn = ((1-all_preds_test)*(1-all_labels_test)).sum()
our_spec = tn / (1-all_labels_test).sum() if (1-all_labels_test).sum()>0 else 0

radar_models = [
    ('Our DenseNet169',      '#FFD700', [our_auc, our_kappa, our_acc, our_sens, our_spec]),
    ('EfficientNet-B4',      '#A855F7', [0.947, 0.78, 0.87, 0.88, 0.86]),
    ('ResNet-152',           '#EC4899', [0.940, 0.76, 0.96, 0.95, 0.97]),
    ('Stanford DenseNet169', '#FF6B6B', [0.929, 0.71, 0.82, 0.84, 0.80]),
]
metric_names = ['AUC', "Cohen's κ", 'Accuracy', 'Sensitivity', 'Specificity']
N = len(metric_names)

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#0d1117')
ax = fig.add_subplot(111, polar=True)

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
for name, color, vals in radar_models:
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', lw=2.5, ms=7, label=name, color=color)
    ax.fill(angles, v, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_names, fontsize=10, color='white', fontweight='bold')
ax.set_yticklabels([])
ax.set_ylim(0, 1.1)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8,
          facecolor='#1a1a2e', edgecolor='#555', labelcolor='white')
ax.set_facecolor('#0d1117')
ax.set_title('Model Performance Radar', fontsize=14, color='white', fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/radar_chart.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('✅ CELL 10 complete — Radar chart saved')


# %% ═══════════════════════════════════════════════════════════════════════
# ════════  RAG INTEGRATION — CELLS 11-17  ════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
#
# After training DenseNet169 (Cells 1-10), these cells:
#   11. Load MedEmbed (1024-dim) + Cross-Encoder reranker
#   12. Upload bone fracture PDFs → page-by-page extraction → chunking
#   13. Embed chunks + store in Weaviate (vector + BM25)
#   14. Hybrid retrieval + reranking functions
#   15. X-ray context injection (model predictions → enrich RAG query)
#   16. Full integrated pipeline: X-ray → fracture detection → RAG → diagnosis
#   17. Interactive text-only clinical query interface


# %% ─────────────────────────────────────────────────────────────────────
# 🧬 CELL 11 — Load MedEmbed + Cross-Encoder Reranker (Persistent)
# ─────────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, CrossEncoder
import weaviate
from weaviate.embedded import EmbeddedOptions
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# ── RAG Config ────────────────────────────────────────────────────────
CHUNK_SIZE     = 500     # chars per chunk
CHUNK_OVERLAP  = 100     # overlap between chunks
TOP_K_RETRIEVE = 20      # candidates from Weaviate
RERANKER_TOP_K = 5       # final chunks sent to LLM
HYBRID_ALPHA   = 0.6     # 0.6 vector + 0.4 BM25
WEAVIATE_CLASS = "BoneFractureKnowledge"

print(f"⏳ Starting Weaviate embedded instance with persistence at: {WEAVIATE_DIR}")
try:
    wv_client = weaviate.connect_to_embedded(
        version="1.24.0",
        options=EmbeddedOptions(
            port=8090, 
            grpc_port=50060,
            persistence_data_path=WEAVIATE_DIR # <--- This enables persistence
        ),
        environment_variables={"ENABLE_MODULES": "text2vec-transformers"},
    )
except Exception as e:
    print(f"⚠️ Port conflict detected ({e}), trying another port...")
    wv_client = weaviate.connect_to_embedded(
        version="1.24.0",
        options=EmbeddedOptions(
            port=8091, 
            grpc_port=50061,
            persistence_data_path=WEAVIATE_DIR
        ),
        environment_variables={"ENABLE_MODULES": "text2vec-transformers"},
    )
    
print(f"✅ Weaviate ready: {wv_client.is_ready()}")
embed_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1", device=str(DEVICE))
print(f"✅ MedEmbed loaded — dim: {embed_model.get_sentence_embedding_dimension()}")

print("⏳ Loading Cross-Encoder reranker …")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device=str(DEVICE))
print("✅ Cross-Encoder loaded")

def embed_texts(texts, batch_size=32):
    return embed_model.encode(texts, batch_size=batch_size,
                               normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)

def embed_query(query):
    return embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

print(f"\n  Chunk size  : {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP}")
print(f"  Retrieve K  : {TOP_K_RETRIEVE} → Rerank → {RERANKER_TOP_K}")
print(f"  Hybrid α    : {HYBRID_ALPHA} vector / {1-HYBRID_ALPHA:.1f} BM25")



# %% ─────────────────────────────────────────────────────────────────────
# 📄 CELL 12 — Upload PDFs → Page-by-Page Extraction → Chunking
# ─────────────────────────────────────────────────────────────────────────
# Each PDF is processed PAGE BY PAGE using PyMuPDF (fitz).
# Each page is then CHUNKED into 500-char pieces with 100-char overlap.
# Each chunk gets metadata:
# {
#   "chunk_id":    "radiology_handbook_p87_c2",
#   "text":        "Hairline fractures appear as thin radiolucent lines...",
#   "source_file": "radiology_handbook.pdf",
#   "page_number": 87,
#   "chunk_index": 2,
#   "body_part":   "wrist",         ← auto-tagged via keyword detection
#   "ingested_at": "2026-03-17T..."
# }

BODY_PART_KEYWORDS = {
    "wrist":    ["wrist", "radius", "ulna", "carpal", "scaphoid", "colles"],
    "elbow":    ["elbow", "humerus", "olecranon", "radial head"],
    "finger":   ["finger", "phalange", "phalanx", "metacarpal"],
    "shoulder": ["shoulder", "clavicle", "glenohumeral", "rotator"],
    "forearm":  ["forearm", "radial shaft", "ulnar shaft"],
    "hand":     ["hand", "metacarpal", "hamate"],
    "humerus":  ["humerus", "proximal humerus", "humeral"],
}

def detect_body_part(text):
    t = text.lower()
    for part, kws in BODY_PART_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return part
    return "general"

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()

def load_and_chunk_pdf(pdf_path):
    """
    Page-by-page PDF extraction + chunk-by-chunk splitting.
    Output: list of chunk dicts with full metadata JSON.
    """
    filename = os.path.basename(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = fitz.open(pdf_path)
    chunks_out = []
    print(f"  📄 {filename}: {len(doc)} pages")

    for page_idx in range(len(doc)):
        raw  = doc[page_idx].get_text("text")
        text = clean_text(raw)
        if len(text) < 50:
            continue
        for ci, chunk_text in enumerate(splitter.split_text(text)):
            if len(chunk_text.strip()) < 30:
                continue
            chunks_out.append({
                "chunk_id":     f"{filename.replace('.pdf','')}_p{page_idx}_c{ci}",
                "text":         chunk_text.strip(),
                "source_file":  filename,
                "page_number":  page_idx,
                "chunk_index":  ci,
                "chunk_length": len(chunk_text),
                "body_part":    detect_body_part(chunk_text),
                "document_type":"clinical_guide",
                "ingested_at":  datetime.utcnow().isoformat() + "Z",
            })
    doc.close()
    return chunks_out

def load_txt_docx(file_path):
    import docx2txt
    filename = os.path.basename(file_path)
    raw = open(file_path, encoding="utf-8", errors="ignore").read() \
          if file_path.endswith(".txt") else docx2txt.process(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [{"chunk_id": f"{filename}_c{i}", "text": c.strip(),
             "source_file": filename, "page_number": 0, "chunk_index": i,
             "chunk_length": len(c), "body_part": detect_body_part(c),
             "document_type": "document", "ingested_at": datetime.utcnow().isoformat()+"Z"}
            for i, c in enumerate(splitter.split_text(clean_text(raw))) if len(c.strip()) >= 30]

# %% ─────────────────────────────────────────────────────────────────────
# 📄 CELL 12 — Persistent PDF/Doc Ingestion
# ─────────────────────────────────────────────────────────────────────────
# Each PDF is processed PAGE BY PAGE and cached in Google Drive.
# If files already exist in DOC_DIR, they are automatically loaded.

def process_and_add_docs(file_list, doc_dir, existing_chunks):
    new_chunks = []
    for filename in file_list:
        dest = os.path.join(doc_dir, filename)
        # Check if already in chunks to avoid duplicates
        if any(c['source_file'] == filename for c in existing_chunks):
            print(f"  ⏩ {filename} already ingested.")
            continue
            
        print(f"  📄 Processing {filename}...")
        chunks = load_and_chunk_pdf(dest) if filename.lower().endswith(".pdf") \
                 else load_txt_docx(dest)
        new_chunks.extend(chunks)
        print(f"     → {len(chunks)} chunks extracted")
    return new_chunks

# 1. Check existing files in Drive
existing_docs = [f for f in os.listdir(DOC_DIR) if f.lower().endswith(('.pdf', '.txt', '.docx'))]

# PRESERVE LOADED STATE: Only init if empty
if 'all_chunks' not in globals():
    all_chunks = []

if existing_docs:
    print(f"📂 Found {len(existing_docs)} existing documents in Drive.")
    # Filter out what's already in all_chunks
    already_ingested = {c['source_file'] for c in all_chunks}
    to_process = [f for f in existing_docs if f not in already_ingested]
    
    if to_process:
        print(f"⏳ Processing {len(to_process)} new/missing documents...")
        new_chunks = process_and_add_docs(to_process, DOC_DIR, all_chunks)
        all_chunks.extend(new_chunks)
    else:
        print("✅ All Drive documents are already in the current session state.")

# 2. Allow uploading NEW documents
print("\n📂 Upload NEW bone fracture documents (or cancel to skip):")
uploaded_docs = colab_files.upload()

if uploaded_docs:
    new_files = []
    for filename, data in uploaded_docs.items():
        dest = os.path.join(DOC_DIR, filename)
        with open(dest, "wb") as f: f.write(data)
        new_files.append(filename)
    
    new_chunks = process_and_add_docs(new_files, DOC_DIR, all_chunks)
    all_chunks.extend(new_chunks)

print(f"\n📊 Total chunks prepared: {len(all_chunks)}")


# %% ─────────────────────────────────────────────────────────────────────
# 🗄️ CELL 13 — Persistent Weaviate Storage + Timing Experiment
# ─────────────────────────────────────────────────────────────────────────
t_start_wv = time.time()
collection = wv_client.collections.get(WEAVIATE_CLASS)

# Check if collection exists and has data
try:
    obj_count = collection.aggregate.over_all(total_count=True).total_count
except:
    obj_count = 0

if obj_count > 0:
    print(f"✅ DRIVE HIT: Weaviate collection '{WEAVIATE_CLASS}' found with {obj_count} existing objects.")
    record_time("Weaviate DB Initialization (Load)", t_start_wv)
    TIME_TRACKER["Weaviate_Status"] = f"Loaded from Drive ({obj_count} objects)"
else:
    print(f"⏳ DRIVE MISS: Weaviate collection empty/missing. Initializing schema...")
    if wv_client.collections.exists(WEAVIATE_CLASS):
        wv_client.collections.delete(WEAVIATE_CLASS)
        
    collection = wv_client.collections.create(
        name=WEAVIATE_CLASS,
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128, max_connections=64,
        ),
        properties=[
            Property(name="chunk_id",      data_type=DataType.TEXT),
            Property(name="text",          data_type=DataType.TEXT),
            Property(name="source_file",   data_type=DataType.TEXT),
            Property(name="page_number",   data_type=DataType.INT),
            Property(name="chunk_index",   data_type=DataType.INT),
            Property(name="chunk_length",  data_type=DataType.INT),
            Property(name="body_part",     data_type=DataType.TEXT),
            Property(name="document_type", data_type=DataType.TEXT),
            Property(name="ingested_at",   data_type=DataType.TEXT),
        ],
    )
    
    if all_chunks:
        print(f"\n🧬 Embedding {len(all_chunks)} chunks with MedEmbed…")
        chunk_texts = [c["text"] for c in all_chunks]
        all_vectors = embed_texts(chunk_texts, batch_size=32)
        
        print(f"⬆️  Uploading to Weaviate for persistence…")
        with collection.batch.fixed_size(batch_size=100) as batch:
            for chunk, vec in zip(all_chunks, all_vectors):
                batch.add_object(
                    properties={k: chunk[k] for k in
                        ["chunk_id","text","source_file","page_number",
                         "chunk_index","chunk_length","body_part","document_type","ingested_at"]},
                    vector=vec.tolist(),
                )
        obj_count = collection.aggregate.over_all(total_count=True).total_count
        print(f"✅ Initialized storage with {obj_count} objects.")
    else:
        print("⚠️ No chunks found to ingest. Please upload documents in Cell 12.")
    
    record_time("Weaviate DB Initialization (Fresh)", t_start_wv)
    TIME_TRACKER["Weaviate_Status"] = "Freshly Indexed (Slow)"


# %% ─────────────────────────────────────────────────────────────────────
# 🔀 CELL 14 — Hybrid Search + Cross-Encoder Reranking Functions
# ─────────────────────────────────────────────────────────────────────────
# hybrid_search():  Weaviate BM25 + vector search combined (alpha-weighted)
# rerank():         Cross-Encoder scores every (query, chunk) pair together

def hybrid_search(query, top_k=TOP_K_RETRIEVE):
    """
    Weaviate hybrid search:
      - Dense: cosine similarity on MedEmbed vectors
      - Sparse: BM25 on raw text
      - Combined with alpha=0.6 (60% vector / 40% BM25)

    Returns candidates list, each:
    {
      "text":        "Hairline fractures appear...",
      "source_file": "radiology.pdf",
      "page_number": 87,
      "body_part":   "wrist",
      "hybrid_score": 0.87
    }
    """
    q_vec = embed_query(query).tolist()
    resp  = collection.query.hybrid(
        query=query,           # BM25 uses this text
        vector=q_vec,          # dense search uses this
        alpha=HYBRID_ALPHA,    # 0.6 vector, 0.4 BM25
        limit=top_k,
        return_metadata=MetadataQuery(score=True, explain_score=True),
    )
    candidates = []
    for obj in resp.objects:
        candidates.append({
            "text":         obj.properties.get("text", ""),
            "source_file":  obj.properties.get("source_file", "unknown"),
            "page_number":  obj.properties.get("page_number", 0),
            "chunk_id":     obj.properties.get("chunk_id", ""),
            "body_part":    obj.properties.get("body_part", "general"),
            "hybrid_score": round(obj.metadata.score, 4) if obj.metadata else 0.0,
        })
    print(f"  🔀 Hybrid search → {len(candidates)} candidates")
    return candidates

def rerank(query, candidates, top_k=RERANKER_TOP_K):
    """
    Cross-Encoder reranking:
      - Reads (query + chunk) TOGETHER through transformer
      - Scores each pair: how well does this chunk answer the query?
      - Sorts by score, keeps top_k

    Output per chunk adds: reranker_score, reranker_rank
    """
    if not candidates: return []
    scores = reranker.predict([(query, c["text"]) for c in candidates],
                              show_progress_bar=False)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:top_k]
    result = []
    for i, (score, cand) in enumerate(ranked):
        cand["reranker_score"] = round(float(score), 4)
        cand["reranker_rank"]  = i + 1
        result.append(cand)

    print(f"  ⚖️  Reranked → top {len(result)} chunks:")
    for c in result:
        print(f"     #{c['reranker_rank']}  {c['source_file']} p{c['page_number']+1}"
              f"  | reranker: {c['reranker_score']:.4f} | hybrid: {c['hybrid_score']:.4f}")
    return result

print("✅ CELL 14 ready — hybrid_search() and rerank() loaded")


# %% ─────────────────────────────────────────────────────────────────────
# 🩻 CELL 15 — X-ray Context Injection + LLM Diagnosis Generation
# ─────────────────────────────────────────────────────────────────────────
# Takes DenseNet169 + Gemini trauma JSON → builds enriched RAG query
# Then generates structured clinical JSON via Groq/Gemini LLM

def build_rag_query(user_question, model_output, gemini_json=None):
    """
    Merge model predictions + Gemini fracture JSON into enriched query.

    Input:
      model_output = {"body_part": "WRIST", "prediction": "Abnormal", "confidence": 0.91}
      gemini_json  = {"fracture_types": {"displaced": {"detected": True, "location": "distal radius"}}}

    Output (enriched query sent to Weaviate):
      "[X-RAY CONTEXT] Body Part: WRIST | Prediction: Abnormal (91%) |
       Fracture: Displaced (loc: distal radius) |
       [CLINICAL QUESTION] What treatment is recommended?"
    """
    parts = [
        f"Body Part: {model_output.get('body_part','unknown').upper()}",
        f"Prediction: {model_output.get('prediction','unknown')} "
        f"({model_output.get('confidence',0):.0%} confidence)",
    ]
    if gemini_json:
        tf = gemini_json.get("trauma_findings", {})
        bf = tf.get("bone_fracture", {})
        detected_fx = [
            f"{name.replace('_fracture','').replace('_',' ')} fracture "
            f"(loc: {info.get('location','N/A')}, conf: {info.get('confidence',0):.0%})"
            for name, info in bf.get("fracture_types", {}).items()
            if info.get("detected")
        ]
        if detected_fx:
            parts.append("Fracture types: " + ", ".join(detected_fx))
        dis = tf.get("dislocation", {})
        if dis.get("detected"):
            parts.append(f"Dislocation: {dis.get('joint_affected','joint')} detected")

    enriched = f"[X-RAY CONTEXT] {' | '.join(parts)}\n[CLINICAL QUESTION] {user_question}"
    print(f"\n  📝 Enriched query:\n     {enriched[:200]}…")
    return enriched

DIAGNOSIS_PROMPT = """\
You are an expert clinical radiologist AI. Use ONLY the provided context.

RETRIEVED MEDICAL KNOWLEDGE:
{context}

CLINICAL QUESTION:
{question}

Return ONLY valid JSON (no markdown fences):
{{
  "diagnosis":             "<primary diagnosis>",
  "confidence":            <float 0-1>,
  "clinical_significance": "<what this finding means>",
  "normal_reference":      "<normal radiographic values>",
  "recommended_tests":     ["<test1>", "<test2>"],
  "treatment_protocol":    "<management plan>",
  "red_flags":             ["<warning1>", "<warning2>"],
  "sources_used":          ["<file p.page>"],
  "summary":               "<one paragraph clinical summary>"
}}"""

def build_context_string(reranked_chunks):
    parts = []
    for i, c in enumerate(reranked_chunks, 1):
        parts.append(
            f"[Source {i}: {c['source_file']} | Page {c['page_number']+1} "
            f"| Reranker Score: {c.get('reranker_score','N/A')}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

def generate_diagnosis(question, reranked_chunks):
    """Call Groq (free) or Gemini text → return structured JSON diagnosis."""
    context = build_context_string(reranked_chunks)
    prompt  = DIAGNOSIS_PROMPT.format(context=context, question=question)
    sources = [f"{c['source_file']} p.{c['page_number']+1}" for c in reranked_chunks]

    # Try Groq first (free, 14,400 req/day)
    try:
        from langchain_groq import ChatGroq
        llm  = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=GROQ_API_KEY)
        text = llm.invoke(prompt).content
    except Exception as e:
        print(f"  ⚠️ Groq failed ({e}), trying Gemini text…")
        if USE_GEMINI:
            text = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt).text
        else:
            return {"error": "Both Groq and Gemini unavailable.", "sources_used": sources}

    t = text.strip()
    if t.startswith("```"): t = t.split("\n",1)[1].rsplit("```",1)[0]
    try:
        result = json.loads(t)
    except json.JSONDecodeError:
        result = {"raw_response": t}
    result["sources_used"] = sources
    return result

print("✅ CELL 15 ready — build_rag_query() and generate_diagnosis() loaded")


# %% ─────────────────────────────────────────────────────────────────────
# 🔗 CELL 16 — Full Integrated Pipeline: X-ray → RAG → Diagnosis
# ─────────────────────────────────────────────────────────────────────────
# Upload an X-ray → runs all 8 steps:
#   1. DenseNet169 inference (abnormality prob + body part)
#   2. GradCAM heatmap
#   3. Gemini Vision trauma JSON (fracture types, dislocation)
#   4. Query enrichment (model output merged into query)
#   5. MedEmbed → 1024-dim query vector
#   6. Weaviate hybrid search (BM25 + vector) → 20 candidates
#   7. Cross-Encoder reranking → top 5
#   8. Groq/Gemini LLM → structured JSON diagnosis

xray_transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MURA_MEAN, MURA_STD),
])

def run_full_pipeline(xray_path, user_question="What fracture type is present and how should it be treated?"):
    print("=" * 65)
    print(f"🦴 FULL BONE FRACTURE RAG PIPELINE")
    print(f"   Image   : {os.path.basename(xray_path)}")
    print(f"   Question: {user_question}")
    print("=" * 65)

    # [1] DenseNet169
    print("\n⏳ [1/8] DenseNet169 inference…")
    pil_gray   = Image.open(xray_path).convert("L")
    img_tensor = xray_transform(pil_gray)
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor.unsqueeze(0).to(DEVICE))).item()
    pred = "Abnormal" if prob >= THRESHOLD else "Normal"
    model_output = {"body_part": "Unknown", "prediction": pred, "confidence": prob}
    print(f"  ✅ {pred} ({prob:.2%})")

    # [2] GradCAM
    print("⏳ [2/8] GradCAM heatmap…")
    cam_overlay, cam_raw = get_gradcam_heatmap(model, img_tensor, pil_gray, DEVICE)

    # [3] Gemini Vision
    print("⏳ [3/8] Gemini Vision trauma analysis…")
    gemini_json = None
    if USE_GEMINI and gemini_client:
        gemini_json = gemini_trauma_analysis(xray_path, gemini_client, GEMINI_MODEL)
        if gemini_json:
            model_output["body_part"] = gemini_json.get("body_part_detected", "Unknown")
            print(f"  ✅ Gemini: {gemini_json.get('overall_abnormality','?')} "
                  f"| body: {model_output['body_part']} "
                  f"| conf: {gemini_json.get('confidence',0):.2f}")

    # [4] Enrich query
    print("⏳ [4/8] Enriching RAG query…")
    enriched_query = build_rag_query(user_question, model_output, gemini_json)

    # [5-6] Hybrid search
    print("⏳ [5/8] MedEmbed + Weaviate hybrid search…")
    candidates = hybrid_search(enriched_query, top_k=TOP_K_RETRIEVE)

    # [7] Rerank
    print(f"⏳ [6/8] Cross-Encoder reranking → top {RERANKER_TOP_K}…")
    reranked = rerank(enriched_query, candidates, top_k=RERANKER_TOP_K)

    # [8] LLM diagnosis
    print("⏳ [7/8] Generating clinical diagnosis (LLM)…")
    diagnosis = generate_diagnosis(enriched_query, reranked)

    # [8] Visualize 4-panel result
    print("⏳ [8/8] Rendering result…")
    orig_np = np.array(pil_gray.resize((IMG_SIZE, IMG_SIZE)))
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f"🦴 {os.path.basename(xray_path)} | DenseNet169: {pred} ({prob:.2%})",
        fontsize=14, fontweight='bold', color='white'
    )

    axes[0].imshow(orig_np, cmap='gray')
    axes[0].set_title("Original X-ray", fontsize=11, color='white', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(cam_overlay)
    axes[1].set_title("GradCAM Heatmap", fontsize=11, color='white', fontweight='bold')
    axes[1].axis('off')

    # Annotated
    ann = orig_np.copy()
    ann_bgr = cv2.cvtColor(ann, cv2.COLOR_GRAY2BGR)
    hot_mask = cam_raw > 0.5
    if hot_mask.any():
        h_s = ann.shape[0] / cam_raw.shape[0]
        w_s = ann.shape[1] / cam_raw.shape[1]
        ys, xs = np.where(hot_mask)
        x1,y1 = int(xs.min()*w_s)-8, int(ys.min()*h_s)-8
        x2,y2 = int(xs.max()*w_s)+8, int(ys.max()*h_s)+8
        box_c = (0,0,255) if prob >= THRESHOLD else (0,200,0)
        cv2.rectangle(ann_bgr, (max(0,x1),max(0,y1)),
                      (min(319,x2),min(319,y2)), box_c, 2)
        badge = f"{'ABNORMAL' if prob>=THRESHOLD else 'NORMAL'} {prob:.0%}"
        cv2.putText(ann_bgr, badge, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_c, 1)
    axes[2].imshow(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Detection Overlay", fontsize=11, color='white', fontweight='bold')
    axes[2].axis('off')

    # Diagnosis card
    axes[3].set_facecolor('#0d1117'); axes[3].axis('off')
    lines = [
        f"{'─'*36}",
        f"Diagnosis  : {diagnosis.get('diagnosis','N/A')[:30]}",
        f"Confidence : {diagnosis.get('confidence','N/A')}",
        f"{'─'*36}",
        f"Treatment:",
        f"  {str(diagnosis.get('treatment_protocol','N/A'))[:70]}",
        f"{'─'*36}",
        "Red Flags:",
    ] + [f"  ⚠️ {r}" for r in diagnosis.get('red_flags',[])[:3]] + [
        f"{'─'*36}", "Sources:",
    ] + [f"  📄 {s}" for s in diagnosis.get('sources_used',[])[:5]]
    axes[3].text(0.04, 0.96, "\n".join(lines),
                 transform=axes[3].transAxes, fontsize=8.5,
                 verticalalignment='top', fontfamily='monospace',
                 color='white', linespacing=1.6,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a2e',
                           edgecolor='#555', alpha=0.95))
    axes[3].set_title("RAG Clinical Diagnosis", fontsize=11, color='white', fontweight='bold')

    for ax in axes: ax.tick_params(colors='white')
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/rag_result_{os.path.basename(xray_path)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()

    # Save full result JSON
    full_result = {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "image_path":      xray_path,
        "user_question":   user_question,
        "enriched_query":  enriched_query,
        "xray_analysis":   {"densenet169": model_output, "gemini_vision": gemini_json},
        "retrieved_sources": reranked,
        "diagnosis":       diagnosis,
    }
    json_path = f"{OUTPUT_DIR}/last_rag_result.json"
    with open(json_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)

    print(f"\n  💾 Saved: {out_path}")
    print(f"  💾 JSON : {json_path}")
    print(f"\n{'━'*65}")
    print("✅ Pipeline complete!\n")
    print("🩺 DIAGNOSIS JSON:")
    print(json.dumps(diagnosis, indent=2))
    return full_result

# ── Upload X-ray and run ──────────────────────────────────────────────
print("📂 Upload an X-ray image (PNG/JPG):")
xray_uploaded = colab_files.upload()
for xray_name, xray_data in xray_uploaded.items():
    xray_path = os.path.join(XRAY_DIR, xray_name)
    with open(xray_path, "wb") as f: f.write(xray_data)
    result = run_full_pipeline(xray_path)


# %% ─────────────────────────────────────────────────────────────────────
# 💬 CELL 17 — Interactive Text-Only Clinical Query Interface
# ─────────────────────────────────────────────────────────────────────────
# No X-ray upload needed — just type a clinical question

def ask_bone_rag(question):
    """Text-only RAG: embed question → hybrid search → rerank → LLM."""
    print(f"\n{'='*65}\n❓ QUESTION: {question}\n{'='*65}")
    candidates = hybrid_search(question, top_k=TOP_K_RETRIEVE)
    reranked   = rerank(question, candidates, top_k=RERANKER_TOP_K)
    diagnosis  = generate_diagnosis(question, reranked)
    print("\n🩺 STRUCTURED DIAGNOSIS:")
    print(json.dumps(diagnosis, indent=2))
    return diagnosis

SAMPLE_QUESTIONS = [
    "What are the radiographic features of a displaced distal radius fracture?",
    "How does a hairline fracture appear on X-ray versus a comminuted fracture?",
    "What is the treatment for a greenstick fracture in children?",
    "What red flags indicate a pathological bone fracture?",
    "When is surgical fixation preferred over cast immobilization for wrist fractures?",
]

print("🦴 Sample clinical questions:\n")
for i, q in enumerate(SAMPLE_QUESTIONS, 1):
    print(f"  [{i}] {q}")
print()

# Run a sample
ask_bone_rag(SAMPLE_QUESTIONS[0])


# %% ─────────────────────────────────────────────────────────────────────
# 📊 CELL 19 — Persistence Experiment Results
# ─────────────────────────────────────────────────────────────────────────
print("📈 PERSISTENCE PERFORMANCE SUMMARY")
print("━" * 40)
print(f"{'Step':<30} | {'Status':<30} | {'Time (s)':<10}")
print("━" * 40)

steps = ["Dataset Retrieval", "Model Initialization", "Weaviate DB Initialization"]
for step in steps:
    # Try to find the matching key in tracker
    time_key = next((k for k in TIME_TRACKER if step in k), None)
    status_key = f"{step.split()[0]}_Status"
    
    time_val = f"{TIME_TRACKER[time_key]:.2f}" if time_key else "N/A"
    status_val = TIME_TRACKER.get(status_key, "N/A")
    
    print(f"{step:<30} | {status_val:<30} | {time_val:<10}")

print("━" * 40)
if MODE == "DRIVE":
    total_saved = sum(v for k, v in TIME_TRACKER.items() if isinstance(v, (int, float)))
    print(f"✅ Total Setup Time: {total_saved:.2f} seconds")
    print("💡 Tip: In MANUAL mode, Dataset download and Model training would take ~5-10 minutes.")

# %% ─────────────────────────────────────────────────────────────────────
# 🔒 CELL 20 — Close Weaviate
# ─────────────────────────────────────────────────────────────────────────
wv_client.close()
print("✅ Weaviate connection closed.")

