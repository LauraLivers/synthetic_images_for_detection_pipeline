# created by Gemini 3.1 Pro
import os
import shutil
from pathlib import Path

def consolidate_images(src_path: str, dest_path: str):
    src_dir = Path(src_path)
    dest_dir = Path(dest_path)
    
    # Create the destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist. Please check the path.")
        return

    # Valid image extensions
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    
    copied_count = 0
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = Path(root) / file
            
            # Check if it's an image
            if file_path.suffix.lower() in valid_extensions:
                # Get the name of the immediate parent folder (e.g. 'lambda0')
                parent_folder_name = file_path.parent.name
                
                # Create a new unique name
                # Example: "image123.jpg" in folder "lambda0" -> "image123_lambda0.jpg"
                new_filename = f"{file_path.stem}_{parent_folder_name}{file_path.suffix}"
                new_file_path = dest_dir / new_filename
                
                # Copy the file
                shutil.copy2(file_path, new_file_path)
                copied_count += 1
                
                print(f"Copied: {file_path.name} -> {new_filename}")

    print(f"\nConsolidation complete. Copied {copied_count} images to {dest_dir}")

if __name__ == "__main__":
    SRC_BASE = "/Users/laura/Desktop/ba_thesis/Laura/MoBI_outputs"
    DEST = "/Users/laura/Desktop/ba_thesis/Laura/MoBI_outputs/consolidated_synthetic_images"
    
    # We will search for folders starting with 'inpaint_results_lambda' in the source base path
    import glob
    folders = glob.glob(os.path.join(SRC_BASE, "inpaint_results_lambda*"))
    for folder in folders:
        print(f"Processing folder: {folder}")
        consolidate_images(folder, DEST)
