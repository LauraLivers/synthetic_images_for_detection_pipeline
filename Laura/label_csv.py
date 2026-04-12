"""create .csv file mapping the filenames to their label index. 
Find index with python -c "from ultralytics import YOLO; model = YOLO('yolov8m-oiv7.pt'); print({v: k for k, v in model.names.items() if '[label]' in v})
or run yolo_classes.py"""
import os
import csv
import sys
import argparse

def create_label_csv(input_path, class_index):
    ladder_folder = os.path.join(input_path, 'ladder')
    no_ladder_folder = os.path.join(input_path, 'no_ladder')
    output_path = os.path.join(os.path.dirname(__file__), f'image_mapped_labels_{class_index}.csv')
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_name', 'ladder'])
        if os.path.isdir(ladder_folder):
            for fname in os.listdir(ladder_folder):
                fpath = os.path.join(ladder_folder, fname)
                if os.path.isfile(fpath) and fname.endswith(('.jpg', '.jpeg', '.png')):
                    writer.writerow([fname, class_index])
        if os.path.isdir(no_ladder_folder):
            for fname in os.listdir(no_ladder_folder):
                fpath = os.path.join(no_ladder_folder, fname)
                if os.path.isfile(fpath) and fname.endswith(('.jpg', '.jpeg', '.png')):
                    writer.writerow([fname, 0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--class_index', type=int, required=True)
    args = parser.parse_args()
    create_label_csv(args.input_path, args.class_index)