# 1. Install required packages
# NOTE: every time your Colab runtime disconnects and reconnects to a new VM,
# these installs are wiped and must run again — this is expected, not an error.
!pip install -q pymupdf
!pip install -q nougat-ocr
!pip install -q transformers==4.38.2
!pip install -q albumentations==1.3.1

import io
import os
import time
import fitz  # PyMuPDF
from PIL import Image
import torch

# --- MOUNT GOOGLE DRIVE — critical for surviving disconnects ---
# Local Colab disk (/content/...) is wiped every time the runtime disconnects
# and reallocates to a new VM. Google Drive is the only thing that persists.
from google.colab import drive
drive.mount('/content/drive')

# --- REDIRECT HUGGINGFACE CACHE TO DRIVE ---
# This must be set BEFORE importing nougat/transformers, so all their internal
# downloads go straight to Drive and persist across disconnects. Using the
# library's own from_pretrained() download path (instead of a manual
# snapshot_download into a raw folder) avoids the checkpoint key-mismatch bug
# that caused "weights newly initialized" — from_pretrained handles the
# sharding/index resolution correctly on its own.
os.environ["HF_HOME"] = "/content/drive/MyDrive/OnesPeer/hf_cache"

from nougat import NougatModel
from nougat.postprocessing import markdown_compatible

# --- CONFIG ---
# Put your PDF inside Drive (e.g. upload it to My Drive/OnesPeer/ first)
pdf_path = "/content/drive/MyDrive/OnesPeer/11 SL CH 03 Motion in a Straight Line.pdf"
output_filename = "/content/drive/MyDrive/OnesPeer/sla_11_ch3.mmd"
progress_dir = "/content/drive/MyDrive/OnesPeer/nougat_page_outputs"
BATCH_SIZE = 4   # drop to 2 if you hit CUDA OOM

# --- HELPER FUNCTIONS ---
def move_to_device(obj, dev):
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, dev) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, dev) for v in obj]
    return obj

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# --- PIPELINE START ---
doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"📄 Loaded PDF: {pdf_path} ({total_pages} total pages)")

print("🖼️ Rasterizing PDF pages into PIL images...")
pages_pil = []
for page in doc:
    pix = page.get_pixmap(dpi=96)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    pages_pil.append(img)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading Nougat Model on {device.upper()} (downloading/caching to Drive on first run)...")
model, loading_info = NougatModel.from_pretrained("facebook/nougat-small", output_loading_info=True)

# Sanity check: catches the silent-garbage bug where checkpoint fails to load
missing = loading_info.get("missing_keys", [])
if len(missing) > 5:
    raise RuntimeError(
        f"❌ {len(missing)} weight tensors failed to load from checkpoint "
        f"(model would run on random weights and produce garbage). "
        f"Do not proceed — first few missing keys: {missing[:5]}"
    )
print(f"✅ Checkpoint loaded correctly ({len(missing)} missing keys, expected 0)")

if device == "cuda":
    model = model.half()
model = model.to(device)
model.eval()

# --- RESUME SUPPORT: skip pages already saved from a prior interrupted run ---
os.makedirs(progress_dir, exist_ok=True)
already_done = {
    int(f.split("_")[1].split(".")[0])
    for f in os.listdir(progress_dir) if f.startswith("page_")
}
if already_done:
    print(f"🔁 Resuming — {len(already_done)} pages already saved, skipping those.")

remaining_indices = [i for i in range(len(pages_pil)) if i not in already_done]

print(f"⏳ Processing {len(remaining_indices)} remaining page(s)...\n")
page_times = []

for batch_idx, indices in enumerate(chunk(remaining_indices, BATCH_SIZE)):
    t0 = time.time()

    batch_imgs = [pages_pil[idx] for idx in indices]
    tensors = [model.encoder.prepare_input(img, random_padding=False) for img in batch_imgs]
    batch_tensor = torch.stack(tensors, dim=0)
    if device == "cuda":
        batch_tensor = batch_tensor.half()
    batch_tensor = move_to_device(batch_tensor, device)

    with torch.no_grad():
        output = model.inference(image_tensors=batch_tensor)

    elapsed = time.time() - t0
    page_times.append(elapsed)

    # SAVE EACH PAGE IMMEDIATELY — this is the key change
    for j, idx in enumerate(indices):
        page_text = markdown_compatible(output["predictions"][j])
        page_path = os.path.join(progress_dir, f"page_{idx:04d}.mmd")
        with open(page_path, "w", encoding="utf-8") as pf:
            pf.write(page_text)

    print(f"  ✅ Batch {batch_idx+1} (pages {indices[0]+1}-{indices[-1]+1}) "
          f"— {elapsed:.1f}s ({elapsed/len(indices):.1f}s/page) — saved to disk")

print(f"\n✅ All pages processed. Assembling final file...")

# --- ASSEMBLE FINAL FILE FROM SAVED PAGES ---
final_parts = []
for i in range(len(pages_pil)):
    page_path = os.path.join(progress_dir, f"page_{i:04d}.mmd")
    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as pf:
            final_parts.append(pf.read())
    else:
        final_parts.append(f"[MISSING PAGE {i+1}]")
        print(f"⚠️ Page {i+1} missing from disk — was skipped or failed")

with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n\n".join(final_parts))

print(f"\n🎉 SUCCESS! Output saved as: {output_filename}")
print(f"📊 Verify: {os.path.getsize(output_filename)} bytes")
