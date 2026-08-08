import time
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# 1. Double check that this file name EXACTLY matches the file inside your folder!
pdf_filename = "11 SL CH 01 Physical World.pdf" 
output_filename = "sl_arora_output.md"

print("⚙️ Configuring pipeline options (Turning off visual OCR)...")
# Tell docling to skip the buggy OCR step and extract native text
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

# Pack this rule into the PDF settings
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

print(f"📄 Starting high-speed conversion for: {pdf_filename}")
start_time = time.time()

try:
    # 2. Convert the book using our custom settings
    result = converter.convert(pdf_filename)
    
    # 3. Export the beautiful markdown
    markdown_content = result.document.export_to_markdown()
    
    # 4. Save to your laptop
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        
    end_time = time.time()
    print("---")
    print(f"🎉 Success! The book has been converted.")
    print(f"📁 Saved file as: {output_filename}")
    print(f"⏳ Time taken: {end_time - start_time:.2f} seconds")

except Exception as e:
    print(f"❌ An error occurred: {e}")