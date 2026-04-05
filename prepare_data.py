import os
import shutil
source_dir = r""    #Old dataset Location
dest_dir = r""  #New created dataset location
mapping = {
    "biological": "wet",
    "paper": "dry",
    "cardboard": "dry",
    "trash": "dry",
    "clothes": "dry",
    "shoes": "dry",
    "plastic": "recyclable",
    "metal": "recyclable",
    "glass": "recyclable",
    "battery": "biohazardous"
}

print("Creating new folder structure...")
for new_class in set(mapping.values()):
    os.makedirs(os.path.join(dest_dir, new_class), exist_ok=True)

print("Copying and organizing images...")
for old_class, new_class in mapping.items():
    old_path = os.path.join(source_dir, old_class)
    new_path = os.path.join(dest_dir, new_class)
    
    if os.path.exists(old_path):
        for filename in os.listdir(old_path):
            src_file = os.path.join(old_path, filename)
            dst_file = os.path.join(new_path, f"{old_class}_{filename}") 
            shutil.copy(src_file, dst_file)
        print(f"Mapped {old_class} -> {new_class}")
    else:
        print(f"Warning: Folder {old_class} not found in source directory.")
print("Dataset reorganized successfully!")