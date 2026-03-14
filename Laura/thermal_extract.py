""" run with input path in termial. """

import os
from PIL import Image

def extract_thermal_images(input_path):
    output_path = os.path.join(input_path, 'thermal')
    os.makedirs(output_path, exist_ok=True)
    for fname in os.listdir(input_path):
        fpath = os.path.join(input_path, fname)
        if os.path.isfile(fpath):
            try:
                with Image.open(fpath) as img:
                    if img.size == (320, 240): # unique format for thermal images
                        os.rename(fpath, os.path.join(output_path, fname))
            except Exception:
                pass  

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python thermal_extract.py <input_folder>')
        sys.exit(1)
    extract_thermal_images(sys.argv[1])