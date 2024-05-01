from ultralytics import YOLO

# Load a model
model = YOLO('240501.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
if __name__ == '__main__':
    results = model.train(data='./data.yaml', epochs=200, imgsz=640, device=[0], patience=50, batch= -1)