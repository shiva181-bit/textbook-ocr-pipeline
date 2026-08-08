import os
import time
import fitz  # This is PyMuPDF
from google import genai
from google.genai import types

# 1. API KEY ROTATION SETUP
# Add multiple keys here (even from different Google accounts) to multiply your speed!
API_KEYS = [
    "YOUR_FIRST_GEMINI_API_KEY",
    # "YOUR_SECOND_GEMINI_API_KEY", 
]

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
print(f"🔑 Loaded {len(API_KEYS)} API Key(s) for rotation optimization.")
print(f"⚡ Running high-speed pipeline...")

# 3. The Optimized Automation Loop
for page_num in range(start_page, end_page):
    actual_page_display = page_num + 1
    file_name = f"page_{actual_page_display:03d}.md"
    file_path = os.path.join(output_folder, file_name)
    
    # Smart Resume check
    if os.path.exists(file_path):
        print(f"⏩ Skipping Page {actual_page_display} (Already exists).")
        continue

    # Pick the current API key based on the page number index
    current_key_index = page_num % len(API_KEYS)
    active_key = API_KEYS[current_key_index]
    
    # Initialize the specific client for this request
    client = genai.Client(api_key=active_key)
    
    success = False
    
    while not success:
        try:
            print(f"⏳ Processing Page {actual_page_display}/{end_page} using Key [{current_key_index}]...")
                
            page = doc[page_num]
            
            # TOKEN OPTIMIZATION: Convert to Grayscale at 150 DPI to slash token sizes
            pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
            image_bytes = pix.tobytes("jpeg")
            
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[system_prompt, image_part]
            )
            
            if response.text:
                markdown_text = response.text.strip()
            else:
                markdown_text = f""
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
                
            print(f"✅ Saved: {file_path}")
            success = True
            
            # Pacing delay depends on how many keys you have
            # If you have 2 keys, a 6-second sleep is perfectly safe and fast.
            pacing_delay = 12 / len(API_KEYS)
            time.sleep(max(pacing_delay, 2))
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "503" in error_msg:
                print(f"⚠️ Key [{current_key_index}] throttled. Waiting 20 seconds for lock clearance...")
                time.sleep(20)
            else:
                print(f"❌ Structural error on Page {actual_page_display}: {error_msg}")
                # Save an error file so it doesn't break the entire automation batch
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"")
                break 

print("\n🎉 Pipeline execution finalized.")