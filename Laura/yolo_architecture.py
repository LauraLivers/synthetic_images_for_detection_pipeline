from ultralytics import YOLO
model = YOLO("yolov8m-oiv7.pt")
print(model.model)