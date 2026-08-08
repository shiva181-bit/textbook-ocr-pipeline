# 1. Install PyMuPDF (fitz) if not already installed
!pip install -q pymupdf

import io
import os
import time
import fitz  # PyMuPDF
from PIL import Image
import torch
from huggingface_hub import snapshot_download
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible

# --- CONFIG ---
SAMPLE_PAGES = 5        # set to None to process the whole PDF
BATCH_SIZE = 4           # try 4 first on a T4 (16GB); drop to 2 if you hit OOM
pdf_path = "11 SL CH 03 Motion in a Straight Line.pdf"
output_filename = "sla_11_ch3_sample.mmd"

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
# 2. Download Model Checkpoint Safely
print("📥 Checking for Nougat model files...")
model_dir = snapshot_download(repo_id="facebook/nougat-small", local_dir="./nougat_weights")

# 3. Render PDF pages using PyMuPDF
doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"📄 Loaded PDF: {pdf_path} ({total_pages} total pages)")

print("🖼️ Rasterizing PDF pages into PIL images...")
pages_pil = []
page_range = range(total_pages) if SAMPLE_PAGES is None else range(min(SAMPLE_PAGES, total_pages))
for i in page_range:
    page = doc[i]
    pix = page.get_pixmap(dpi=96)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    pages_pil.append(img)

print(f"🔎 Running on {len(pages_pil)} page(s) this run (SAMPLE_PAGES={SAMPLE_PAGES})")

# 4. Load Nougat Model on GPU in fp16
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading Nougat Model on {device.upper()} from local directory...")
model = NougatModel.from_pretrained(model_dir)
if device == "cuda":
    model = model.half()          # fp16 cast BEFORE .to(device)
model = model.to(device)
model.eval()

# 5. Process pages in batches, generate LaTeX Markdown
print("⏳ Extracting text and math formulas...\n")
predictions = [None] * len(pages_pil)
page_times = []
avg_running = None

for batch_idx, indices in enumerate(chunk(list(range(len(pages_pil))), BATCH_SIZE)):
    t0 = time.time()

    batch_imgs = [pages_pil[idx] for idx in indices]
    tensors = [
        model.encoder.prepare_input(img, random_padding=False)
        for img in batch_imgs
    ]
    batch_tensor = torch.stack(tensors, dim=0)
    if device == "cuda":
        batch_tensor = batch_tensor.half()
    batch_tensor = move_to_device(batch_tensor, device)

    with torch.no_grad():
        output = model.inference(image_tensors=batch_tensor)

    elapsed = time.time() - t0
    per_page = elapsed / len(indices)
    page_times.append(elapsed)
    avg_running = sum(page_times) / len(page_times)
    flag = "⚠️ SLOW BATCH" if elapsed > 2 * avg_running and batch_idx > 0 else ""

    for j, idx in enumerate(indices):
        page_text = output["predictions"][j]
        page_text = markdown_compatible(page_text)
        predictions[idx] = "\n" + page_text

    print(f"  ✅ Batch {batch_idx+1} (pages {indices[0]+1}-{indices[-1]+1}) "
          f"— {elapsed:.1f}s total, {per_page:.1f}s/page {flag}")

total_time = sum(page_times)
print(f"\nTotal: {total_time:.1f}s for {len(pages_pil)} pages "
      f"({total_time/len(pages_pil):.1f}s/page avg)")

if SAMPLE_PAGES is not None and total_pages > SAMPLE_PAGES:
    projected = (total_time / len(pages_pil)) * total_pages
    print(f"📈 Projected time for all {total_pages} pages: "
          f"~{projected/60:.1f} min")

# 6. Save output
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n\n".join(p for p in predictions if p is not None))

print(f"\n🎉 SUCCESS! Output saved as: {output_filename}")
