from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.pt")  # load an official model
model = YOLO("best.pt")  # load a custom model

# Predict with the model
results = model(source=1 ,show=True,save=False,conf=0.6)  # predict on an image