import os
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

# 1. Get the current folder where this script is running
current_dir = os.getcwd()
file_name = "my_paper.pdf"
file_path = os.path.join(current_dir, file_name)

print(f"Current Directory: {current_dir}")
print(f"Looking for: {file_path}")

# 2. Check if the file exists before processing
if not os.path.exists(file_path):
    print("❌ Error: File still not found! Checking for double extension...")
    # Try checking if it was accidentally named .pdf.pdf
    if os.path.exists(file_path + ".pdf"):
        file_path += ".pdf"
        print(f"✅ Found it with double extension: {file_path}")
    else:
        print("📁 Files actually in this folder:")
        print(os.listdir(current_dir))
else:
    print("✅ File found! Extracting text...")
    
    logic_params = LAParams(line_margin=0.5, char_margin=2.0, detect_vertical=True)
    text = extract_text(file_path, laparams=logic_params)
    
    clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    print(clean_text)