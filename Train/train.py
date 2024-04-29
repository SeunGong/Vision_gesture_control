from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
if __name__ == '__main__':
    results = model.train(data='./data.yaml', epochs=1000, imgsz=640, device=[0], patience=200)