import os
import fitz  # This is PyMuPDF
import google.generativeai as genai

# 1. Setup the AI Engine
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)

# Using gemini-1.5-flash for fast, free vision processing
model = genai.GenerativeModel('gemini-1.5-flash')

pdf_path = "11 SL CH 01 Physical World.pdf"
output_md_path = "page_5_vision_test.md"

doc = fitz.open(pdf_path)

# System Instructions focused ONLY on returning raw Markdown and pristine LaTeX
system_prompt = """
You are an elite Physics textbook parser. Your job is to convert the provided textbook page image into clean Markdown text.
Follow these rules strictly:
1. Ignore any background watermarks or text artifacts like "MOHIT SI". Do not include them in the output.
2. Format ALL scientific equations, variables, powers (like 10^-14), Greek symbols (like beta or epsilon), vectors, and fractions using flawless LaTeX notation ($ for inline, $$ for block formulas).
3. Output ONLY the raw markdown text content of the page. 
Do NOT wrap your response in markdown code blocks like ```markdown. Just start typing the content directly.
"""

print(f"📸 Loaded PDF: {pdf_path} ({len(doc)} pages total)")
print("🚀 Testing the Vision Pipeline on Page 5...")

# 2. Extract Page 5 as a high-res image directly in memory
page = doc[4]  # 0-indexed, so 4 is Page 5
pix = page.get_pixmap(dpi=200)  # Clear 200 DPI resolution
image_bytes = pix.tobytes("jpeg")

# 3. Create the payload for the Vision model
image_part = {
    "mime_type": "image/jpeg",
    "data": image_bytes
}

# 4. Run the model and save the raw markdown output
try:
    response = model.generate_content([system_prompt, image_part])
    markdown_text = response.text.strip()
    
    # 5. Save the text straight to a markdown file
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
        
    print("\n🎉 SUCCESS! Raw markdown with perfect LaTeX has been generated.")
    print(f"📁 Saved file as: {output_md_path}")
    print("--- Preview of the output below ---")
    print(markdown_text[:500] + "\n...")

except Exception as e:
    print(f"\n❌ Pipeline Error: {e}")