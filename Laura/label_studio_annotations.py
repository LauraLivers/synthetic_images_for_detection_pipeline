import os

ANNOTATION_DIR = 'robo_images/yolo_annotations/labels'

for filename in os.listdir(ANNOTATION_DIR):
    if filename.endswith('.txt'):
        new_name = filename.split('-', 1)[1]
        os.rename(
            os.path.join(ANNOTATION_DIR, filename),
            os.path.join(ANNOTATION_DIR, new_name)
        )