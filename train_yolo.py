from ultralytics import YOLO

# Încarcă modelul YOLOv8 pre-antrenat
model = YOLO('yolov8n.pt')  

# Antrenează modelul pe datasetul tău
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='food_model'
)
