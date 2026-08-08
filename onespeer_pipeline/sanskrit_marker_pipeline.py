# 1. Install Marker + PyMuPDF
!pip install -q -U marker-pdf
!pip install -q pymupdf

import os
import time
import fitz  # PyMuPDF

# --- MOUNT GOOGLE DRIVE ---
from google.colab import drive
drive.mount('/content/drive')

# --- REDIRECT MODEL CACHE TO DRIVE ---
os.environ["HF_HOME"] = "/content/drive/MyDrive/OnesPeer/hf_cache"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# --- CONFIG ---
pdf_path = "/content/drive/MyDrive/OnesPeer/sanskrit_class6_ch1.pdf"
CHUNK_SIZE = 15   # pages per resumable chunk — consider lowering to 8-10 for
                   # image-heavy children's textbooks, which tend to have more
                   # illustrations per page than the physics/math chapters

pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
output_filename = f"/content/drive/MyDrive/OnesPeer/markdown_output/{pdf_stem}.md"
progress_dir = f"/content/drive/MyDrive/OnesPeer/marker_chunk_outputs/{pdf_stem}"

os.makedirs(progress_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
images_output_dir = os.path.join(progress_dir, "images")
os.makedirs(images_output_dir, exist_ok=True)
local_chunk_pdf = "/content/_chunk_temp.pdf"

# --- MARKER CONFIG — plain local extraction, no LLM/API ---
# "languages" hints Surya's OCR toward Devanagari script, which can improve
# recognition of conjuncts/ligatures common in Sanskrit vs letting it guess.
config = {
    "output_format": "markdown",
    "languages": "hi",   # Devanagari script hint — Sanskrit doesn't have its
                          # own ISO code in Surya's list, Hindi's "hi" uses the
                          # same script and is the standard stand-in
}
config_parser = ConfigParser(config)

# --- LOAD MODELS ONCE ---
print("🚀 Loading Marker models (downloads to Drive cache on first run)...")
t0 = time.time()
model_dict = create_model_dict()
print(f"✅ Models loaded in {time.time()-t0:.1f}s")

converter = PdfConverter(
    config=config_parser.generate_config_dict(),
    artifact_dict=model_dict,
    processor_list=config_parser.get_processors(),
    renderer=config_parser.get_renderer(),
    llm_service=config_parser.get_llm_service(),
)

# --- SPLIT SOURCE PDF INTO CHUNKS ---
doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"📄 Loaded PDF: {pdf_path} ({total_pages} total pages)")

chunk_ranges = [
    (i, min(i + CHUNK_SIZE, total_pages))
    for i in range(0, total_pages, CHUNK_SIZE)
]
print(f"📦 Split into {len(chunk_ranges)} chunk(s) of up to {CHUNK_SIZE} pages each")

# --- RESUME SUPPORT ---
already_done = {
    int(f.split("_")[1].split("-")[0])
    for f in os.listdir(progress_dir) if f.startswith("chunk_")
}
if already_done:
    print(f"🔁 Resuming — {len(already_done)} chunk(s) already saved, skipping those.")

# --- PROCESS EACH CHUNK ---
chunk_times = []
for start, end in chunk_ranges:
    if start in already_done:
        continue

    t0 = time.time()

    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
    chunk_doc.save(local_chunk_pdf)
    chunk_doc.close()

    rendered = converter(local_chunk_pdf)
    text, _, images = text_from_rendered(rendered)

    # Save extracted images (illustrations), converting to RGB to avoid the
    # Pillow JPEG-encoder bug; one bad image won't crash the whole chunk.
    if images:
        os.makedirs(images_output_dir, exist_ok=True)
        for img_name, img_obj in images.items():
            prefixed_name = f"p{start:04d}_{img_name}"
            try:
                if img_obj.mode != "RGB":
                    img_obj = img_obj.convert("RGB")
                img_obj.save(os.path.join(images_output_dir, prefixed_name))
                text = text.replace(img_name, prefixed_name)
            except Exception as e:
                print(f"  ⚠️ Failed to save image {img_name}: {e} — leaving placeholder text")
                text = text.replace(
                    f"![]({img_name})", f"[image extraction failed: {img_name}]"
                )

    os.makedirs(progress_dir, exist_ok=True)
    chunk_path = os.path.join(progress_dir, f"chunk_{start:04d}-{end:04d}.md")
    with open(chunk_path, "w", encoding="utf-8") as f:
        f.write(text)

    elapsed = time.time() - t0
    chunk_times.append(elapsed)
    pages_in_chunk = end - start
    print(f"  ✅ Pages {start+1}-{end} — {elapsed:.1f}s "
          f"({elapsed/pages_in_chunk:.1f}s/page) — saved to disk")

if chunk_times:
    avg = sum(chunk_times) / len(chunk_times)
    print(f"\nAvg chunk time: {avg:.1f}s")

# --- ASSEMBLE FINAL FILE FROM SAVED CHUNKS, INLINING IMAGES AS BASE64 ---
print("\n✅ Assembling final markdown file (embedding images as base64)...")
import re
import base64
import mimetypes

final_parts = []
for start, end in chunk_ranges:
    chunk_path = os.path.join(progress_dir, f"chunk_{start:04d}-{end:04d}.md")
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            final_parts.append(f.read())
    else:
        final_parts.append(f"[MISSING PAGES {start+1}-{end}]")
        print(f"⚠️ Chunk {start+1}-{end} missing — was skipped or failed")

combined_text = "\n\n".join(final_parts)

def embed_image(match):
    alt_text, img_filename = match.group(1), match.group(2)
    img_path = os.path.join(images_output_dir, img_filename)
    if not os.path.exists(img_path):
        return match.group(0)
    mime_type, _ = mimetypes.guess_type(img_path)
    mime_type = mime_type or "image/jpeg"
    with open(img_path, "rb") as img_f:
        b64_data = base64.b64encode(img_f.read()).decode("utf-8")
    return f"![{alt_text}](data:{mime_type};base64,{b64_data})"

combined_text = re.sub(r"!\[(.*?)\]\(([^)\s]+)\)", embed_image, combined_text)

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(combined_text)

print(f"\n🎉 SUCCESS! Output saved as: {output_filename}")
print(f"📊 Verify: {os.path.getsize(output_filename)} bytes")
