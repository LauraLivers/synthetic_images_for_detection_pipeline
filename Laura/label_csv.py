""" input path given by user. creates .csv file containing all image names with label [0,1]"""

import os
import csv

def create_label_csv(input_path):
    ladder_folder = os.path.join(input_path, 'ladder')
    no_ladder_folder = os.path.join(input_path, 'no_ladder')
    output_path = os.path.join(os.path.dirname(__file__), 'resnet50_labels.csv')
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'ladder'])
        # ladder images
        if os.path.isdir(ladder_folder):
            for fname in os.listdir(ladder_folder):
                fpath = os.path.join(ladder_folder, fname)
                if os.path.isfile(fpath):
                    writer.writerow([fname, 1])
        # no_ladder images
        if os.path.isdir(no_ladder_folder):
            for fname in os.listdir(no_ladder_folder):
                fpath = os.path.join(no_ladder_folder, fname)
                if os.path.isfile(fpath):
                    writer.writerow([fname, 0])

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python label_csv.py <input_folder>')
        sys.exit(1)
    create_label_csv(sys.argv[1])