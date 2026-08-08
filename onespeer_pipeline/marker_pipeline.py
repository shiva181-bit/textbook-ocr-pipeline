# 1. Install Marker + PyMuPDF
# IMPORTANT: if you previously ran the Nougat pipeline in this same Colab session,
# restart the runtime first (Runtime -> Restart session) before running this cell.
!pip install -q -U marker-pdf
!pip install -q pymupdf

import os
import time
import fitz  # PyMuPDF

# --- MOUNT GOOGLE DRIVE — persists progress and model cache across disconnects ---
from google.colab import drive
drive.mount('/content/drive')

# --- REDIRECT MODEL CACHE TO DRIVE (avoids re-downloading Marker's models every session) ---
os.environ["HF_HOME"] = "/content/drive/MyDrive/OnesPeer/hf_cache"

# --- GEMINI API KEY — required for AI-generated image descriptions ---
# Get a free key at https://aistudio.google.com/apikey and paste it below.
os.environ["GOOGLE_API_KEY"] = "PASTE_YOUR_GEMINI_API_KEY_HERE"

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

# --- CONFIG ---
pdf_path = "/content/drive/MyDrive/OnesPeer/11 SL CH 03 Motion in a Straight Line.pdf"
CHUNK_SIZE = 15   # pages per resumable chunk — tune down if sessions are unstable

# Progress dir and output filename are derived from the PDF's own name,
# so switching pdf_path automatically gets a clean, isolated progress folder —
# no risk of one chapter's leftover chunks being picked up by another.
pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
output_filename = f"/content/drive/MyDrive/OnesPeer/markdown_output/{pdf_stem}.md"
progress_dir = f"/content/drive/MyDrive/OnesPeer/marker_chunk_outputs/{pdf_stem}"

os.makedirs(progress_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_filename), exist_ok=True)
local_chunk_pdf = "/content/_chunk_temp.pdf"  # ephemeral, regenerated each run — fine to lose

# --- MARKER CONFIG: replace images with AI-generated text descriptions ---
# instead of extracting them as separate files. use_llm asks Gemini to
# describe each diagram/figure in words; disable_image_extraction means no
# separate image files are produced — the description is written directly
# into the markdown text where the image would have been. Result: one fully
# self-contained .md file, no external image dependencies.
config_parser = ConfigParser({
    "output_format": "markdown",
    "use_llm": True,
    "disable_image_extraction": True,
})

# --- LOAD MODELS ONCE ---
print("🚀 Loading Marker models (downloads to Drive cache on first run)...")
t0 = time.time()
model_dict = create_model_dict()
print(f"✅ Models loaded in {time.time()-t0:.1f}s")

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

    # Build a temporary single-chunk PDF
    chunk_doc = fitz.open()
    chunk_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
    chunk_doc.save(local_chunk_pdf)
    chunk_doc.close()

    # Convert with Marker — images become LLM-written text descriptions,
    # inserted inline in the markdown. No separate image files produced.
    converter = PdfConverter(
        artifact_dict=model_dict,
        config=config_parser.generate_config_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(local_chunk_pdf)
    text, _, images = text_from_rendered(rendered)

    # Save immediately to Drive — text only, fully self-contained
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

# --- ASSEMBLE FINAL FILE FROM SAVED CHUNKS ---
print("\n✅ Assembling final markdown file...")
final_parts = []
for start, end in chunk_ranges:
    chunk_path = os.path.join(progress_dir, f"chunk_{start:04d}-{end:04d}.md")
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            final_parts.append(f.read())
    else:
        final_parts.append(f"[MISSING PAGES {start+1}-{end}]")
        print(f"⚠️ Chunk {start+1}-{end} missing — was skipped or failed")

with open(output_filename, "w", encoding="utf-8") as f:
    f.write("\n\n".join(final_parts))

print(f"\n🎉 SUCCESS! Output saved as: {output_filename}")
print(f"📊 Verify: {os.path.getsize(output_filename)} bytes")
