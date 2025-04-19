from ultralytics import YOLO

model = YOLO(r"F:\YOLO\0209_classify\fb8320-main\fb8320-main\Classification\yolov8s-cls.pt")
model.train(data='F:/YOLO//0209_classify/YOLOv8_Finally/dataset/', epochs=200, batch=16, imgsz=640)
