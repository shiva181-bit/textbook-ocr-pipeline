# textbook-ocr-pipeline
# Academic PDF → Markdown/LaTeX Conversion Pipeline

Automated pipeline for converting large academic PDF corpora (physics, mathematics, and Devanagari-script textbooks) into structured Markdown with embedded LaTeX mathematical notation — built to run reliably on free-tier Google Colab despite frequent runtime disconnects and GPU quota limits.

This repo includes **every version of the pipeline written during development** — both the final working scripts and the earlier failed/abandoned attempts — kept together intentionally rather than cleaned up, so the dead ends are visible rather than erased. Several of the failures here are non-obvious and worth documenting for anyone hitting the same walls.

Working and non-working scripts are not separated into different folders; all conversion scripts live together in one folder. Each script's own header comments indicate whether it was a working version or an abandoned attempt, and why.

---

## Additional Deliverables

- Full project report
- Presentation slides

---

## Why This Exists

Academic PDFs — especially math and physics texts — are hostile to naive text extraction: equations need LaTeX, diagrams need to stay associated with the surrounding content, and a 100+ page chapter needs to survive an unreliable free-tier GPU session without losing an hour of compute to a single disconnect. This pipeline was built specifically to solve the *reliability* problem as much as the *conversion* problem.

---

## Architecture (Final Pipeline)

1. **Split** — source PDF is split into fixed-size page chunks (default: 15 pages) using PyMuPDF.
2. **Convert** — each chunk is run through Marker (Surya OCR/layout/table/equation models) to produce markdown with LaTeX-formatted equations.
3. **Persist immediately** — each chunk's output is written to Google Drive *as soon as it finishes*, not held in memory until the end.
4. **Resume automatically** — on every run, already-completed chunks are detected and skipped, so a runtime disconnect costs at most one partial chunk, never the whole book.
5. **Assemble** — once all chunks for a PDF are done, they're concatenated into a single final `.md` file.

Two persistence layers make this durable across Colab's ephemeral local disk:
- **Model cache** → redirected to Drive via `HF_HOME`, so weights aren't re-downloaded every session.
- **Chunk output + final markdown** → written directly to Drive, never to local `/content/` storage.

---

## Setup

```bash
pip install -q "marker-pdf<2.0.0" "surya-ocr<0.20.0"
pip install -q pymupdf
```

> **Version pinning is intentional, not optional.** `marker-pdf` 2.0+ pulls in Surya 2, which requires a `vllm` server or a compiled `llama.cpp` binary — neither available in a stock Colab environment.

Run in Google Colab with a GPU runtime (`Runtime → Change runtime type → T4 GPU`). Mount Google Drive when prompted — all persistence depends on it.

---

## Usage

Each working script has a config block near the top (PDF path, chunk size). Update it to point at your source PDF and run. Output path, progress folder, and images folder are all derived automatically from the PDF's filename, so switching books only requires changing the one path variable.

Just re-run the same cell if the Colab session disconnects — it picks up exactly where it left off rather than starting over.

---

## Image Handling

The pipeline saves extracted figures as **separate image files** with plain relative markdown links (`![](book_images/p0000_figure1.jpeg)`), not embedded as base64. This was a deliberate choice after testing base64 embedding: while base64 makes a single self-contained file, it bloats file size significantly and is actively counterproductive if the markdown is meant for ingestion into an LLM/RAG pipeline, since base64 text carries no usable semantic information to a text-only ingestion step. Use `strip_base64_images.py` if you have older output that needs cleaning up.

---
