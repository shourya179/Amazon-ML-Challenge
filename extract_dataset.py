import zipfile
import os

zip_path = r"C:\Users\Shourya\Downloads\6ab10eb3b23ba_student_resource.zip"
extract_dir = r"c:\Users\Shourya\Desktop\dataset"

print(f"Checking if {zip_path} exists...")
if not os.path.exists(zip_path):
    print("Zip file does not exist at specified path!")
else:
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction complete!")
    print("Contents of extract_dir:")
    for root, dirs, files in os.walk(extract_dir):
        rel_root = os.path.relpath(root, extract_dir)
        if rel_root == ".":
            print("Root files/dirs:", dirs, files)
