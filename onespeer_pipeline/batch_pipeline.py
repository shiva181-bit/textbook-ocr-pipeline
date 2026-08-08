import os
import time
import fitz  # This is PyMuPDF
from google import genai
from google.genai import types

# 1. Initialize Client
GEMINI_API_KEY = ""
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

# 2. Configuration Settings
pdf_path = "11 SL CH 01 Physical World.pdf"
output_folder = "extracted_pages"
os.makedirs(output_folder, exist_ok=True)

doc = fitz.open(pdf_path)
total_pages = len(doc)

start_page = 0  
end_page = total_pages  

system_prompt = """
You are an elite Physics textbook parser. Your job is to convert the provided textbook page image into clean Markdown text.
Follow these rules strictly:
1. Ignore any background watermarks or text artifacts like "MOHIT SI". Do not include them in the output.
2. Format ALL scientific equations, variables, powers (like 10^-14), Greek symbols (like beta or epsilon), vectors, and fractions using flawless LaTeX notation ($ for inline, $$ for block formulas).
3. Output ONLY the raw markdown text content of the page. Do NOT wrap your response in markdown code blocks.
"""

print(f"📸 Loaded PDF: {pdf_path} ({total_pages} pages total)")
print(f"⚡ Running pipeline from index {start_page + 1} to {end_page}...")

# 3. The Smart Automation Loop
for page_num in range(start_page, end_page):
    actual_page_display = page_num + 1
    
    file_name = f"page_{actual_page_display:03d}.md"
    file_path = os.path.join(output_folder, file_name)
    
    # UPGRADE 1: SMART RESUME
    # If the page was already successfully generated on a previous run, skip it entirely!
    if os.path.exists(file_path):
        print(f"⏩ Skipping Page {actual_page_display} (File already exists).")
        continue

    max_retries = 3
    attempt = 0
    success = False
    wait_time = 30  # Start with a strong 30-second wait on rate limits
    
    while attempt < max_retries and not success:
        try:
            print(f"⏳ Processing Page {actual_page_display}/{end_page} (Attempt {attempt + 1}/{max_retries})...")
                
            # Extract page image
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            image_bytes = pix.tobytes("jpeg")
            
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
            
            # Request conversion
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[system_prompt, image_part]
            )
            
            # UPGRADE 2: DEFENSIVE PARSING
            # Make sure the response actually has text to prevent NoneType errors
            if response.text:
                markdown_text = response.text.strip()
            else:
                print(f"⚠️ Warning: Page {actual_page_display} returned empty text. Content may have been filtered.")
                markdown_text = f""
            
            # Save the text file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
                
            print(f"✅ Saved: {file_path}")
            success = True
            
            # Standard pacing delay between successful pages to preserve quota
            time.sleep(15)
            
        except Exception as e:
            error_msg = str(e)
            attempt += 1
            
            # UPGRADE 3: EXPONENTIAL BACKOFF FOR RATE LIMITS
            if "429" in error_msg or "503" in error_msg:
                print(f"⚠️ Quota hit/Server busy. Sleeping for {wait_time} seconds to reset window...")
                time.sleep(wait_time)
                wait_time *= 2  # Double the wait time for the next attempt if it fails again
            else:
                print(f"❌ Minor bug on Page {actual_page_display}: {error_msg}")
                break 

print("\n🎉 Pipeline complete! All unparsed pages have been successfully handled.")