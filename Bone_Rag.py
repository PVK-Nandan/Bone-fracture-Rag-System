# ============================================================
# 🦴 Bone Fracture RAG System — Workflow & Architecture
# ============================================================
#
# Step 1 — Document Loading
#
# Purpose: Extract text from documents.
#
# Tools used:
#   pypdf
#   docx2txt
#
# Example output:
# {
#  "source": "radiology_handbook.pdf",
#  "page": 87,
#  "text": "Hairline fractures appear as thin radiolucent lines in cortical bone..."
# }
#
#
# Step 2 — Text Chunking
#
# Large documents are split into smaller pieces.
#
# Tool: LangChain TextSplitter
# Parameters:
#   chunk_size = 500 tokens
#   chunk_overlap = 100 tokens
#
# Example chunk:
# {
#  "chunk_id": "chunk_1023",
#  "text": "Hairline fractures appear as thin radiolucent lines in cortical bone..."
# }
#
#
# Step 3 — Metadata Extraction
#
# "metadata": {
#  "fracture_type": "hairline",
#  "body_part": "wrist",
#  "source": "radiology_handbook",
#  "page": 87
# }
#
#
# Step 4 — Store in Database
#
# {
#  "id": "chunk_1023",
#  "embedding": [0.213, -0.551, 0.882, ...],
#  "text": "Hairline fractures appear as thin radiolucent lines in cortical bone...",
#  "metadata": {
#    "fracture_type": "hairline",
#    "body_part": "wrist",
#    "source": "radiology_handbook",
#    "page": 87
#  }
# }
#
#
# ─────────────────────────────────────
# Bone Fracture Detection (Online Inference)
# ─────────────────────────────────────
#
# DenseNet169 (trained on MURA dataset)
# Detect normal vs abnormal bone
#
# Example output:
# {
#  "body_part": "wrist",
#  "prediction": "abnormal",
#  "confidence": 0.86
# }
#
# Query Generation for RAG:
# {
#  "query": "wrist fracture x-ray radiographic features"
# }
#
# Query Embedding:
# [0.202, -0.501, 0.899, ...]
#
# Vector Similarity Search:
#   query_vector  vs  stored_document_vectors
#   Similarity metric: Cosine similarity
#
#   Chunk        Score
#   chunk_1023   0.89
#   chunk_441    0.85
#   chunk_218    0.81
#
#   Threshold: similarity > 0.80
#   Top-K results selected: K = 3
#
# Retrieved Context:
# {
#  "text": "Hairline fractures appear as thin radiolucent lines in cortical bone...",
#  "metadata": {
#    "fracture_type": "hairline",
#    "body_part": "wrist"
#  },
#  "score": 0.89
# }
#
# LLM Diagnosis:
#   X-ray image + retrieved medical context + model prediction
#
# Prompt example:
#   You are a radiologist.
#   Context: Hairline fractures appear as thin radiolucent lines in cortical bone.
#   Analyze the X-ray image and determine if a fracture is present.
#
# Output:
# {
#  "diagnosis": "fracture",
#  "confidence": 0.92,
#  "explanation": "The X-ray shows a hairline fracture in the distal radius...",
#  "recommendation": "Immobilization with splint for 4-6 weeks"
# }
#
#
# ─────────────────────────────────────
# OFFLINE PIPELINE (Knowledge Base)
# ─────────────────────────────────────
#
# Medical Documents
#        ↓
# Text Extraction
#        ↓
# Chunking
#        ↓
# Metadata Extraction
#        ↓
# Embedding Model (MedEmbed)
#        ↓
# Vector Database (Weaviate)
#
# ─────────────────────────────────────
# ONLINE PIPELINE (Inference)
# ─────────────────────────────────────
#
# User X-ray
#        ↓
# DenseNet169 (MURA)
#        ↓
# GradCAM
#        ↓
# Query Generation
#        ↓
# Embedding Model
#        ↓
# Vector Search
#        ↓
# Retrieve Medical Knowledge
#        ↓
# Gemini Vision
#        ↓
# Final Diagnosis
#
# ─────────────────────────────────────
# Steps Summary:
# ─────────────────────────────────────
#  1. Knowledge base creation
#  2. Document loading
#  3. Chunking
#  4. Metadata extraction
#  5. Embedding generation
#  6. Vector database storage
#  7. Bone fracture detection (DenseNet169 + MURA)
#  8. GradCAM explanation
#  9. Query generation
# 10. Vector similarity search
# 11. Top-K retrieval
# 12. Gemini Vision reasoning
# 13. Final diagnosis output
#
# ============================================================
# 🦴 Bone Fracture RAG System — Google Colab Notebook
#
# Pipeline:
#   Medical PDFs → Chunking → MedEmbed → Weaviate DB
#   User X-ray   → DenseNet169 → GradCAM → Query Generation
#                → Vector Search → Gemini Vision → Diagnosis
#
# Embedding Model : abhinand/MedEmbed-large-v0.1 (1024-dim)
# Vector DB       : Weaviate (embedded, no cloud account needed)
# Bone Model      : DenseNet169 (trained on MURA)
# LLM + Vision    : Gemini 2.0 Flash
# ============================================================


# %% [markdown]
# # 🦴 Bone Fracture RAG System
# **Full Pipeline:**
# ```
# Medical PDFs → Text Extraction → Chunking → Metadata → MedEmbed → Weaviate
#                                                                        ↑
# User X-ray → DenseNet169 → GradCAM → Query → Vector Search ──────────┘
#                                                    ↓
#                                          Retrieve Medical Context
#                                                    ↓
#                                          Gemini Vision + Context
#                                                    ↓
#                                          Final Structured Diagnosis
# ```


# %% ─────────────────────────────────────────────────────────────────────
# CELL 1 ⬇  Install ALL dependencies
# ─────────────────────────────────────────────────────────────────────────
import subprocess, sys

packages = [
    # ── LangChain ───────────────────────────────────────────────────────
    "langchain",
    "langchain-community",
    "langchain-core",
    "langchain-text-splitters",
    "langchain-google-genai",
    "langchain-groq",

    # ── Weaviate (embedded — no cloud account required) ──────────────────
    "weaviate-client>=4.0.0",

    # ── Embedding model ──────────────────────────────────────────────────
    "sentence-transformers",
    "transformers",
    "torch",
    "huggingface-hub",
    "timm",

    # ── Document loaders ─────────────────────────────────────────────────
    "pypdf",
    "python-docx",
    "docx2txt",
    "pymupdf",            # fitz — for page-by-page PDF extraction

    # ── Image processing ─────────────────────────────────────────────────
    "Pillow",
    "opencv-python-headless",

    # ── GradCAM ──────────────────────────────────────────────────────────
    "grad-cam",

    # ── Gemini ───────────────────────────────────────────────────────────
    "google-genai",
    "google-generativeai",

    # ── Monitoring / reliability ─────────────────────────────────────────
    "tenacity",
    "scikit-learn",
    "tqdm",
]

print("⏳ Installing all dependencies …")
for p in packages:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", p, "-q"],
        capture_output=True, text=True
    )
    status = "✅" if r.returncode == 0 else "⚠️"
    print(f"  {status}  {p}")

print("\n✅ All dependencies installed.")


# %% ─────────────────────────────────────────────────────────────────────
# CELL 2 ⬇  Imports, API Keys & Global Config
# ─────────────────────────────────────────────────────────────────────────
import os, json, io, time, re, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm.notebook import tqdm
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
warnings.filterwarnings("ignore")

from google.colab import userdata, files as colab_files
from google import genai
from google.genai import types as gtypes

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

import weaviate
from weaviate.classes.config import (
    Configure, Property, DataType, VectorDistances
)
from weaviate.classes.query import MetadataQuery

# ── Determinism ───────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────────────
DOC_DIR    = "/content/bone_docs"
OUTPUT_DIR = "/content/bone_rag_outputs"
XRAY_DIR   = "/content/xrays"
MODEL_CKPT = "/content/mura_outputs/best_densenet169_mura.pth"  # from your MURA notebook

os.makedirs(DOC_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(XRAY_DIR,   exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────
CHUNK_SIZE      = 500    # tokens per chunk
CHUNK_OVERLAP   = 100    # token overlap between chunks
TOP_K_RETRIEVE  = 20     # candidates from Weaviate
RERANKER_TOP_K  = 5      # final chunks sent to LLM after reranking
IMG_SIZE        = 320    # MURA input size
THRESHOLD       = 0.5    # DenseNet169 abnormality threshold
HYBRID_ALPHA    = 0.6    # 0.6 vector / 0.4 BM25 in Weaviate hybrid search

WEAVIATE_CLASS  = "BoneFractureKnowledge"

# ── API Keys ──────────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
except Exception:
    GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

try:
    GROQ_API_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ── Gemini client ────────────────────────────────────────────────────
GEMINI_MODEL = "models/gemini-2.0-flash"
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    _test = gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents="Reply: OK"
    )
    USE_GEMINI = "OK" in _test.text
    print(f"✅ Gemini connected → {GEMINI_MODEL}")
except Exception as e:
    print(f"⚠️  Gemini unavailable: {e}")
    USE_GEMINI = False
    gemini_client = None

print(f"\n{'━'*55}")
print(f"  Device      : {DEVICE}")
print(f"  Chunk size  : {CHUNK_SIZE} tokens | Overlap: {CHUNK_OVERLAP}")
print(f"  Retrieve K  : {TOP_K_RETRIEVE} → Rerank → {RERANKER_TOP_K}")
print(f"  Hybrid α    : {HYBRID_ALPHA} (vector) / {1-HYBRID_ALPHA:.1f} (BM25)")
print(f"  Gemini      : {'ENABLED ✅' if USE_GEMINI else 'DISABLED ⚠️'}")
print(f"{'━'*55}")


# %% ─────────────────────────────────────────────────────────────────────
# CELL 3 ⬇  Load Embedding Model (MedEmbed-large-v0.1)
# ─────────────────────────────────────────────────────────────────────────
# MedEmbed is a SentenceTransformer fine-tuned on PubMed / medical text
# → produces 1024-dimensional L2-normalized vectors.
# Used for BOTH document chunks and query embedding (symmetric retrieval).

print("⏳ Loading MedEmbed-large-v0.1 … (downloads ~1.3 GB first time)")
embed_model = SentenceTransformer(
    "abhinand/MedEmbed-large-v0.1",
    device=str(DEVICE),
)
print(f"✅ MedEmbed loaded — embedding dim: {embed_model.get_sentence_embedding_dimension()}")

# ── Cross-Encoder Reranker ────────────────────────────────────────────
# Reads (query + chunk) together → fine-grained relevance score
print("⏳ Loading Cross-Encoder reranker …")
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=str(DEVICE),
)
print("✅ Cross-Encoder reranker loaded.")


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts using MedEmbed.
    Returns normalized float32 numpy array of shape (N, 1024).
    """
    return embed_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (1024,) float32 numpy array."""
    return embed_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


# %% ─────────────────────────────────────────────────────────────────────
# CELL 4 ⬇  Document Loading → Chunking → Metadata Extraction
# ─────────────────────────────────────────────────────────────────────────
# STEP 1: Load documents (PDF page by page using PyMuPDF)
# STEP 2: Clean and chunk each page
# STEP 3: Attach metadata per chunk (source, page, body_part)
#
# Output per chunk:
# {
#   "chunk_id":     "radiology_handbook_p87_c2",
#   "text":         "Hairline fractures appear as thin radiolucent lines...",
#   "source_file":  "radiology_handbook.pdf",
#   "page_number":  87,
#   "chunk_index":  2,
#   "chunk_length": 487,
#   "body_part":    "wrist",          ← auto-detected from text / filename
#   "document_type":"clinical_guide",
#   "ingested_at":  "2026-03-16T07:00:00Z"
# }

import fitz  # PyMuPDF

# Body part keyword mapping (auto-tag chunks)
BODY_PART_KEYWORDS = {
    "wrist":    ["wrist", "radius", "ulna", "carpal", "scaphoid", "colles"],
    "elbow":    ["elbow", "humerus", "olecranon", "radial head"],
    "finger":   ["finger", "phalange", "phalanx", "metacarpal", "interphalangeal"],
    "shoulder": ["shoulder", "clavicle", "glenohumeral", "rotator", "acromial"],
    "forearm":  ["forearm", "radial shaft", "ulnar shaft"],
    "hand":     ["hand", "metacarpal", "carpal", "hamate"],
    "humerus":  ["humerus", "proximal humerus", "humeral"],
}

def detect_body_part(text: str) -> str:
    """Auto-detect body part from chunk text using keyword matching."""
    text_lower = text.lower()
    for part, keywords in BODY_PART_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return part
    return "general"


def clean_text(text: str) -> str:
    """Remove excessive whitespace, page artifacts, and control characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)   # strip non-ASCII artifacts
    return text.strip()


def load_and_chunk_pdf(pdf_path: str) -> list[dict]:
    """
    PAGE-BY-PAGE extraction from PDF using PyMuPDF.
    Each page is then CHUNK-BY-CHUNK into 500-token pieces.

    Returns a list of chunk dicts with full metadata.
    """
    filename = os.path.basename(pdf_path)
    chunks_out = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  📄 {filename}: {total_pages} pages")

    for page_idx in range(total_pages):
        raw_text = doc[page_idx].get_text("text")
        cleaned  = clean_text(raw_text)

        if len(cleaned) < 50:          # skip near-empty pages
            continue

        # Split page into chunks
        page_chunks = splitter.split_text(cleaned)

        for chunk_idx, chunk_text in enumerate(page_chunks):
            if len(chunk_text.strip()) < 30:
                continue

            chunk_id   = f"{filename.replace('.pdf','')}_p{page_idx}_c{chunk_idx}"
            body_part  = detect_body_part(chunk_text)

            chunk_obj = {
                "chunk_id":     chunk_id,
                "text":         chunk_text.strip(),
                "source_file":  filename,
                "page_number":  page_idx,         # 0-indexed
                "chunk_index":  chunk_idx,
                "chunk_length": len(chunk_text),
                "body_part":    body_part,
                "document_type": "clinical_guide",
                "ingested_at":  datetime.utcnow().isoformat() + "Z",
            }
            chunks_out.append(chunk_obj)

    doc.close()
    return chunks_out


def load_txt_docx(file_path: str) -> list[dict]:
    """Load TXT or DOCX files and chunk (no page metadata)."""
    import docx2txt
    filename = os.path.basename(file_path)

    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    elif file_path.endswith(".docx"):
        raw = docx2txt.process(file_path)
    else:
        return []

    cleaned = clean_text(raw)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    page_chunks = splitter.split_text(cleaned)
    chunks_out = []
    for i, chunk_text in enumerate(page_chunks):
        if len(chunk_text.strip()) < 30:
            continue
        chunks_out.append({
            "chunk_id":     f"{filename}_c{i}",
            "text":         chunk_text.strip(),
            "source_file":  filename,
            "page_number":  0,
            "chunk_index":  i,
            "chunk_length": len(chunk_text),
            "body_part":    detect_body_part(chunk_text),
            "document_type": "document",
            "ingested_at":  datetime.utcnow().isoformat() + "Z",
        })
    return chunks_out


# ── Upload documents ──────────────────────────────────────────────────
print("📂 Upload your medical documents (PDF, TXT, or DOCX):")
uploaded = colab_files.upload()

all_chunks = []
for filename, data in uploaded.items():
    dest = os.path.join(DOC_DIR, filename)
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  ✔  Saved: {filename}  ({len(data)//1024} KB)")

    if filename.lower().endswith(".pdf"):
        chunks = load_and_chunk_pdf(dest)
    elif filename.lower().endswith((".txt", ".docx")):
        chunks = load_txt_docx(dest)
    else:
        print(f"  ⚠️  Unsupported file type: {filename}")
        chunks = []

    all_chunks.extend(chunks)
    print(f"     → {len(chunks)} chunks extracted")

print(f"\n📊 Total chunks ready for embedding: {len(all_chunks)}")

# Preview first chunk
if all_chunks:
    print("\n🔍 Preview — First chunk JSON:")
    print(json.dumps({k: v for k, v in all_chunks[0].items()
                      if k != "embedding"}, indent=2))


# %% ─────────────────────────────────────────────────────────────────────
# CELL 5 ⬇  Embed all Chunks + Store in Weaviate
# ─────────────────────────────────────────────────────────────────────────
# STEP 4: Generate 1024-dim MedEmbed vector for each chunk
# STEP 5: Upsert to Weaviate (vector + metadata + BM25 auto-index)
#
# Weaviate stores each object in this format internally:
# {
#   "id":       "chunk_1023",
#   "vector":   [0.213, -0.551, 0.882, ...],   (1024 floats)
#   "text":     "Hairline fractures...",
#   "metadata": { "source_file": "...", "page_number": 87, ... }
# }

# ── Weaviate embedded (runs locally — no cloud account needed) ─────────
print("⏳ Starting Weaviate embedded instance …")
wv_client = weaviate.connect_to_embedded(
    version="1.24.0",
    environment_variables={"ENABLE_MODULES": "text2vec-transformers"},
)
print(f"✅ Weaviate connected — is_ready: {wv_client.is_ready()}")

# ── Create schema (collection) ─────────────────────────────────────────
if wv_client.collections.exists(WEAVIATE_CLASS):
    wv_client.collections.delete(WEAVIATE_CLASS)
    print(f"  ♻️  Old collection '{WEAVIATE_CLASS}' deleted.")

collection = wv_client.collections.create(
    name=WEAVIATE_CLASS,
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE,
        ef_construction=128,
        max_connections=64,
    ),
    properties=[
        Property(name="chunk_id",      data_type=DataType.TEXT),
        Property(name="text",          data_type=DataType.TEXT),     # BM25 indexed
        Property(name="source_file",   data_type=DataType.TEXT),
        Property(name="page_number",   data_type=DataType.INT),
        Property(name="chunk_index",   data_type=DataType.INT),
        Property(name="chunk_length",  data_type=DataType.INT),
        Property(name="body_part",     data_type=DataType.TEXT),
        Property(name="document_type", data_type=DataType.TEXT),
        Property(name="ingested_at",   data_type=DataType.TEXT),
    ],
)
print(f"✅ Weaviate collection '{WEAVIATE_CLASS}' created.")

# ── Embed all chunks in batches ────────────────────────────────────────
print(f"\n🧬 Embedding {len(all_chunks)} chunks with MedEmbed …")
chunk_texts = [c["text"] for c in all_chunks]
all_vectors = embed_texts(chunk_texts, batch_size=32)  # shape: (N, 1024)
print(f"✅ Embeddings done — shape: {all_vectors.shape}")

# ── Batch upsert into Weaviate (100 objects per batch) ────────────────
BATCH_SIZE_WEAVIATE = 100
print(f"\n⬆️  Uploading to Weaviate in batches of {BATCH_SIZE_WEAVIATE} …")

with collection.batch.fixed_size(batch_size=BATCH_SIZE_WEAVIATE) as batch:
    for i, (chunk, vector) in enumerate(zip(all_chunks, all_vectors)):
        batch.add_object(
            properties={
                "chunk_id":      chunk["chunk_id"],
                "text":          chunk["text"],
                "source_file":   chunk["source_file"],
                "page_number":   chunk["page_number"],
                "chunk_index":   chunk["chunk_index"],
                "chunk_length":  chunk["chunk_length"],
                "body_part":     chunk["body_part"],
                "document_type": chunk["document_type"],
                "ingested_at":   chunk["ingested_at"],
            },
            vector=vector.tolist(),   # list[float] — 1024 values
        )

total_stored = collection.aggregate.over_all(total_count=True).total_count
print(f"\n✅ Weaviate upsert complete — {total_stored} objects stored.")
print(f"   Each object: vector (1024-dim) + text + metadata (source, page, body_part)")


# %% ─────────────────────────────────────────────────────────────────────
# CELL 6 ⬇  Load Bone Fracture Detection Model (DenseNet169 from MURA)
# ─────────────────────────────────────────────────────────────────────────
# This is YOUR trained model from the MURA notebook.
# If you have the checkpoint (.pth) from that notebook use it directly.
# Otherwise a pre-trained DenseNet169 with MURA-style head is built here.

MURA_MEAN = [0.485, 0.456, 0.406]
MURA_STD  = [0.229, 0.224, 0.225]

# ── Model definition ─────────────────────────────────────────────────
class MURAModel(nn.Module):
    """DenseNet169 backbone + custom head (same as MURA training notebook)."""
    def __init__(self, backbone="densenet169", pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.encoder.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.classifier(self.encoder(x)).squeeze(1)


xray_transform = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(MURA_MEAN, MURA_STD),
])

# ── Load checkpoint ───────────────────────────────────────────────────
bone_model = MURAModel("densenet169", pretrained=True).to(DEVICE)

if os.path.exists(MODEL_CKPT):
    ckpt = torch.load(MODEL_CKPT, map_location=DEVICE)
    bone_model.load_state_dict(ckpt["state"])
    print(f"✅ Loaded MURA checkpoint from: {MODEL_CKPT}")
    print(f"   Checkpoint Kappa: {ckpt.get('kappa', 'N/A'):.4f}")
else:
    print("⚠️  No MURA checkpoint found. Using ImageNet-pretrained DenseNet169.")
    print(f"   Expected path: {MODEL_CKPT}")
    print("   Run your MURA training notebook first, or the model will be less accurate.")

bone_model.eval()
print("✅ Bone fracture model ready.")


# %% ─────────────────────────────────────────────────────────────────────
# CELL 7 ⬇  GradCAM + Query Generation from X-ray
# ─────────────────────────────────────────────────────────────────────────
# STEP: Upload X-ray → DenseNet169 inference → GradCAM heatmap
#       → Gemini Vision trauma JSON → Auto-generate RAG query

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

# ── GradCAM heatmap generator ─────────────────────────────────────────
def get_gradcam(model, img_tensor, pil_gray):
    """
    Run GradCAM on the bone model for the given image.
    Returns: (overlay_rgb H×W×3, cam_raw H×W float).
    """
    target_layers = [model.encoder.features.norm5]
    cam = GradCAM(model=model, target_layers=target_layers)
    cam_raw = cam(
        input_tensor=img_tensor.unsqueeze(0).to(DEVICE),
        targets=[BinaryClassifierOutputTarget(1)],
    )[0]

    orig = np.array(pil_gray.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32)
    orig = orig / (orig.max() + 1e-8)

    cam_resized = cv2.resize(cam_raw, (IMG_SIZE, IMG_SIZE))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    xray_rgb    = np.stack([orig] * 3, axis=-1)
    overlay     = ((1 - 0.4) * xray_rgb + 0.4 * heatmap_rgb).clip(0, 1)

    return (overlay * 255).astype(np.uint8), cam_resized

# ── Gemini Vision trauma JSON analysis ────────────────────────────────
TRAUMA_PROMPT = """\
You are a radiologist. Analyze this bone X-ray and return ONLY valid JSON (no markdown fences):

{
  "overall_abnormality": "normal" | "abnormal",
  "body_part": "<string>",
  "confidence": <float 0-1>,
  "fracture_types": {
    "hairline":     {"detected": true|false, "confidence": <float>, "location": "<string>"},
    "displaced":    {"detected": true|false, "confidence": <float>, "location": "<string>"},
    "comminuted":   {"detected": true|false, "confidence": <float>, "location": "<string>"},
    "greenstick":   {"detected": true|false, "confidence": <float>, "location": "<string>"},
    "stress":       {"detected": true|false, "confidence": <float>, "location": "<string>"}
  },
  "dislocation":    {"detected": true|false, "joint": "<string>"},
  "clinical_summary": "<string>"
}"""


def gemini_trauma_json(image_path: str) -> dict | None:
    """Send X-ray to Gemini Vision → return structured trauma JSON."""
    if not USE_GEMINI:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                gtypes.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
                TRAUMA_PROMPT,
            ],
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"  ⚠️  Gemini vision error: {e}")
        return None


def build_rag_query(user_question: str, model_output: dict, gemini_json: dict | None) -> str:
    """
    Merge user question + DenseNet169 output + Gemini trauma JSON
    into an enriched query string for RAG retrieval.

    Example enriched query:
      "[X-RAY CONTEXT] Body Part: WRIST | Prediction: Abnormal (91%) |
       Fracture: Displaced (loc: distal radius, conf: 89%) |
       [CLINICAL QUESTION] What treatment is recommended?"
    """
    context_parts = []

    # DenseNet169 output
    context_parts.append(
        f"Body Part: {model_output.get('body_part', 'unknown').upper()}"
    )
    context_parts.append(
        f"Prediction: {model_output.get('prediction', 'unknown')} "
        f"({model_output.get('confidence', 0):.0%} confidence)"
    )

    # Gemini fracture findings
    if gemini_json:
        fx_types = gemini_json.get("fracture_types", {})
        detected_fx = [
            f"{name} fracture "
            f"(loc: {info.get('location','N/A')}, conf: {info.get('confidence',0):.0%})"
            for name, info in fx_types.items()
            if info.get("detected")
        ]
        if detected_fx:
            context_parts.append("Fracture types: " + ", ".join(detected_fx))
        else:
            context_parts.append("Fracture: none detected")

        dis = gemini_json.get("dislocation", {})
        if dis.get("detected"):
            context_parts.append(f"Dislocation: {dis.get('joint','joint')} detected")

    xray_context = " | ".join(context_parts)

    enriched_query = (
        f"[X-RAY CONTEXT] {xray_context}\n"
        f"[CLINICAL QUESTION] {user_question}"
    )

    print(f"\n  📝 Enriched RAG Query:\n  {enriched_query}\n")
    return enriched_query


# %% ─────────────────────────────────────────────────────────────────────
# CELL 8 ⬇  Weaviate Hybrid Search + Cross-Encoder Reranking
# ─────────────────────────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """
    Weaviate hybrid search = dense vector similarity + BM25 sparse search
    combined with alpha weighting (0.6 vector / 0.4 BM25).

    Returns list of chunk dicts with hybrid scores.

    Weaviate Response per candidate:
    {
      "text":        "Hairline fractures appear as thin radiolucent lines...",
      "source_file": "radiology_handbook.pdf",
      "page_number": 87,
      "body_part":   "wrist",
      "score":       0.87
    }
    """
    query_vector = embed_query(query).tolist()   # embed enriched query

    response = collection.query.hybrid(
        query=query,                             # BM25 uses this text
        vector=query_vector,                     # vector search uses this
        alpha=HYBRID_ALPHA,                      # 0.6 vector, 0.4 BM25
        limit=top_k,
        return_metadata=MetadataQuery(score=True, explain_score=True),
    )

    candidates = []
    for obj in response.objects:
        candidates.append({
            "text":         obj.properties.get("text", ""),
            "source_file":  obj.properties.get("source_file", "unknown"),
            "page_number":  obj.properties.get("page_number", 0),
            "chunk_id":     obj.properties.get("chunk_id", ""),
            "body_part":    obj.properties.get("body_part", "general"),
            "hybrid_score": round(obj.metadata.score, 4) if obj.metadata else 0.0,
        })

    print(f"  🔀 Hybrid search → {len(candidates)} candidates retrieved")
    return candidates


def rerank(query: str, candidates: list[dict],
           top_k: int = RERANKER_TOP_K) -> list[dict]:
    """
    Cross-Encoder reranking.
    Scores each (query, chunk) pair together through a transformer.
    Returns top_k chunks sorted by reranker score (highest = most relevant).
    """
    if not candidates:
        return []

    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    for i, (score, cand) in enumerate(sorted(
        zip(scores, candidates), key=lambda x: x[0], reverse=True
    )[:top_k]):
        cand["reranker_score"] = round(float(score), 4)
        cand["reranker_rank"]  = i + 1

    reranked = sorted(
        [c for c in candidates if "reranker_score" in c],
        key=lambda x: x["reranker_score"],
        reverse=True,
    )

    print(f"  ⚖️  Reranked {len(candidates)} → top {len(reranked)} chunks:")
    for c in reranked:
        print(f"     #{c['reranker_rank']}  {c['source_file']} p{c['page_number']+1}"
              f"  | reranker: {c['reranker_score']:.4f}"
              f"  | hybrid: {c['hybrid_score']:.4f}")
    return reranked


# %% ─────────────────────────────────────────────────────────────────────
# CELL 9 ⬇  Full RAG Chain — Gemini Vision + Retrieved Context → Diagnosis
# ─────────────────────────────────────────────────────────────────────────

DIAGNOSIS_PROMPT_TEMPLATE = """\
You are an expert clinical radiologist AI assistant.
Use ONLY the information in the provided context to answer the question.
Always be evidence-based. If context is insufficient, say so honestly.

RETRIEVED MEDICAL KNOWLEDGE:
{context}

CLINICAL QUESTION:
{question}

Return ONLY valid JSON with this exact structure (no markdown fences):
{{
  "diagnosis":          "<primary diagnosis>",
  "confidence":         <float 0-1>,
  "clinical_significance": "<what this finding means>",
  "normal_reference":   "<normal radiographic values if applicable>",
  "recommended_tests":  ["<test 1>", "<test 2>"],
  "treatment_protocol": "<recommended management>",
  "red_flags":          ["<warning 1>", "<warning 2>"],
  "sources_used":       ["<file p.page>"],
  "summary":            "<one paragraph clinical summary>"
}}"""


def build_context_string(reranked_chunks: list[dict]) -> str:
    """Format top-K reranked chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(reranked_chunks, 1):
        src  = chunk["source_file"]
        page = chunk["page_number"] + 1   # display 1-indexed
        score = chunk.get("reranker_score", "N/A")
        parts.append(
            f"[Source {i}: {src} | Page {page} | Reranker Score: {score}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_diagnosis(question: str, reranked_chunks: list[dict]) -> dict:
    """
    Send reranked context + question to Groq LLM (or Gemini text fallback).
    Returns structured JSON diagnosis.
    """
    context = build_context_string(reranked_chunks)
    prompt  = DIAGNOSIS_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    sources_used = [
        f"{c['source_file']} p.{c['page_number']+1}"
        for c in reranked_chunks
    ]

    # ── Try Groq first (free, fast) ─────────────────────────────────
    try:
        from langchain_groq import ChatGroq
        from tenacity import retry, wait_exponential, stop_after_attempt
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            api_key=GROQ_API_KEY,
        )
        response_text = groq_llm.invoke(prompt).content

    except Exception as groq_err:
        print(f"  ⚠️  Groq failed ({groq_err}), trying Gemini text …")
        if USE_GEMINI:
            response_text = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            ).text
        else:
            return {"error": "Both Groq and Gemini unavailable."}

    # ── Parse JSON from LLM response ─────────────────────────────────
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"raw_response": text, "sources_used": sources_used}

    result["sources_used"] = sources_used
    return result


# %% ─────────────────────────────────────────────────────────────────────
# CELL 10 ⬇  Full Inference Pipeline — Upload X-ray → Final Diagnosis
# ─────────────────────────────────────────────────────────────────────────

def run_bone_rag_pipeline(xray_path: str, user_question: str) -> dict:
    """
    Complete end-to-end Bone Fracture RAG pipeline.

    Steps:
      1. DenseNet169 → abnormality score + body part
      2. GradCAM     → attention heatmap
      3. Gemini Vision → structured fracture JSON
      4. Query enrichment (model + Gemini output merged into query)
      5. MedEmbed → query vector (1024-dim)
      6. Weaviate hybrid search (vector + BM25) → 20 candidates
      7. Cross-Encoder reranking → top 5 chunks
      8. Groq/Gemini LLM → structured JSON diagnosis
      9. Visualize + return full result JSON

    Returns:
      Full result dict with xray_analysis, rag_result, reranked_sources.
    """
    print("=" * 65)
    print(f"🦴 BONE FRACTURE RAG PIPELINE")
    print(f"   Image   : {os.path.basename(xray_path)}")
    print(f"   Question: {user_question}")
    print("=" * 65)

    # ── Step 1: Load + preprocess X-ray ─────────────────────────────
    print("\n⏳ [1/8] Running DenseNet169 inference …")
    pil_gray   = Image.open(xray_path).convert("L")
    img_tensor = xray_transform(pil_gray)

    bone_model.eval()
    with torch.no_grad():
        logit = bone_model(img_tensor.unsqueeze(0).to(DEVICE))
        prob  = torch.sigmoid(logit).item()

    prediction = "Abnormal" if prob >= THRESHOLD else "Normal"
    model_output = {
        "body_part":  "Unknown",     # Gemini will detect this
        "prediction": prediction,
        "confidence": prob,
        "abnormality_prob": prob,
    }
    print(f"  ✅ DenseNet169: {prediction} ({prob:.2%})")

    # ── Step 2: GradCAM ──────────────────────────────────────────────
    print("⏳ [2/8] Generating GradCAM heatmap …")
    cam_overlay, cam_raw = get_gradcam(bone_model, img_tensor, pil_gray)
    print("  ✅ GradCAM generated.")

    # ── Step 3: Gemini Vision trauma JSON ────────────────────────────
    print("⏳ [3/8] Running Gemini Vision trauma analysis …")
    gemini_json = gemini_trauma_json(xray_path)
    if gemini_json:
        model_output["body_part"] = gemini_json.get("body_part", "Unknown")
        print(f"  ✅ Gemini: {gemini_json.get('overall_abnormality','?')} "
              f"| body part: {gemini_json.get('body_part','?')} "
              f"| conf: {gemini_json.get('confidence',0):.2f}")
    else:
        print("  ⚠️  Gemini unavailable — using model output only.")

    # ── Step 4: Enrich query ─────────────────────────────────────────
    print("⏳ [4/8] Enriching RAG query with model findings …")
    enriched_query = build_rag_query(user_question, model_output, gemini_json)

    # ── Step 5–6: Weaviate hybrid search ─────────────────────────────
    print("⏳ [5/8] Embedding enriched query (MedEmbed) …")
    print("⏳ [6/8] Weaviate hybrid search (Vector + BM25) …")
    candidates = hybrid_search(enriched_query, top_k=TOP_K_RETRIEVE)

    # ── Step 7: Cross-Encoder reranking ──────────────────────────────
    print(f"⏳ [7/8] Cross-Encoder reranking → top {RERANKER_TOP_K} …")
    reranked = rerank(enriched_query, candidates, top_k=RERANKER_TOP_K)

    # ── Step 8: LLM diagnosis ─────────────────────────────────────────
    print("⏳ [8/8] Generating structured diagnosis (LLM) …")
    diagnosis = generate_diagnosis(enriched_query, reranked)

    # ── Visualize ────────────────────────────────────────────────────
    orig_np    = np.array(pil_gray.resize((IMG_SIZE, IMG_SIZE)))
    fig, axes  = plt.subplots(1, 3, figsize=(21, 7))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"🦴 Bone Fracture RAG — {os.path.basename(xray_path)}  |  "
        f"DenseNet169: {prediction} ({prob:.2%})",
        fontsize=14, fontweight="bold", color="white",
    )

    axes[0].imshow(orig_np, cmap="gray")
    axes[0].set_title("Original X-ray", fontsize=12, color="white", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_overlay)
    axes[1].set_title("GradCAM Heatmap", fontsize=12, color="white", fontweight="bold")
    axes[1].axis("off")

    # Diagnosis card
    axes[2].set_facecolor("#0d1117")
    axes[2].axis("off")
    diag_text = "\n".join([
        f"{'─'*38}",
        f"Diagnosis   : {diagnosis.get('diagnosis', 'N/A')}",
        f"Confidence  : {diagnosis.get('confidence', 'N/A')}",
        f"{'─'*38}",
        f"Treatment:",
        f"  {diagnosis.get('treatment_protocol', 'N/A')[:80]}",
        f"{'─'*38}",
        f"Red Flags:",
    ] + [f"  ⚠️  {r}" for r in diagnosis.get("red_flags", [])[:3]] + [
        f"{'─'*38}",
        f"Sources:",
    ] + [f"  📄 {s}" for s in diagnosis.get("sources_used", [])[:5]]
    )
    axes[2].text(
        0.05, 0.97, diag_text,
        transform=axes[2].transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace", color="white", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1a1a2e",
                  edgecolor="#555", alpha=0.95),
    )
    axes[2].set_title("RAG Diagnosis", fontsize=12, color="white", fontweight="bold")

    for ax in axes:
        ax.tick_params(colors="white")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"bone_rag_{os.path.basename(xray_path)}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()

    # ── Final result JSON ─────────────────────────────────────────────
    full_result = {
        "timestamp":        datetime.utcnow().isoformat() + "Z",
        "image_path":       xray_path,
        "user_question":    user_question,
        "enriched_query":   enriched_query,
        "xray_analysis": {
            "densenet169":  model_output,
            "gemini_vision": gemini_json,
        },
        "retrieved_sources": reranked,
        "diagnosis":         diagnosis,
    }

    result_path = os.path.join(OUTPUT_DIR, "last_result.json")
    with open(result_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)
    print(f"\n  💾 Full result JSON saved → {result_path}")
    print(f"\n{'━'*65}")
    print("✅ Pipeline complete!\n")
    return full_result


# ── Upload X-ray and run ─────────────────────────────────────────────
print("📂 Upload an X-ray image (PNG/JPG):")
xray_uploaded = colab_files.upload()

for xray_name, xray_data in xray_uploaded.items():
    xray_path = os.path.join(XRAY_DIR, xray_name)
    with open(xray_path, "wb") as f:
        f.write(xray_data)
    print(f"  ✔  Saved X-ray: {xray_name}")

    USER_QUESTION = "What type of fracture is present and what treatment is recommended?"
    final_result = run_bone_rag_pipeline(xray_path, USER_QUESTION)

    print("\n🩺 FINAL DIAGNOSIS JSON:")
    print(json.dumps(final_result["diagnosis"], indent=2))


# %% ─────────────────────────────────────────────────────────────────────
# CELL 11 ⬇  Interactive Query Interface
# ─────────────────────────────────────────────────────────────────────────
# Use this cell to ask text-only clinical questions (no X-ray upload needed)

def ask_bone_rag(question: str) -> dict:
    """
    Text-only RAG query — no X-ray required.
    Directly embeds question → hybrid retrieval → rerank → LLM diagnosis.
    """
    print(f"\n{'='*65}")
    print(f"❓ QUESTION: {question}")
    print(f"{'='*65}")
    candidates = hybrid_search(question, top_k=TOP_K_RETRIEVE)
    reranked   = rerank(question, candidates, top_k=RERANKER_TOP_K)
    diagnosis  = generate_diagnosis(question, reranked)

    print("\n🩺 STRUCTURED DIAGNOSIS:\n")
    print(json.dumps(diagnosis, indent=2))
    print(f"\n{'='*65}")
    return diagnosis


# Sample bone fracture questions
BONE_QUESTIONS = [
    "What are the radiographic features of a displaced distal radius fracture?",
    "What is the treatment for a greenstick fracture in children?",
    "How does a hairline fracture appear on X-ray compared to normal bone?",
    "What red flags indicate a pathological fracture?",
    "What follow-up imaging is needed after a suspected stress fracture?",
]

print("🦴 Sample questions — run any below:\n")
for i, q in enumerate(BONE_QUESTIONS, 1):
    print(f"  [{i}] {q}")

# ── Run one sample ────────────────────────────────────────────────────
ask_bone_rag("What are the radiographic features of a displaced distal radius fracture?")


# %% ─────────────────────────────────────────────────────────────────────
# CELL 12 ⬇  Close Weaviate Connection (run when done)
# ─────────────────────────────────────────────────────────────────────────
# Always close the embedded Weaviate instance before ending the session
wv_client.close()
print("✅ Weaviate connection closed cleanly.")
