from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Single stream with batch-size 1 inference
# source = 'rtsp://example.com/media.mp4'  # RTSP, RTMP, TCP or IP streaming address

# Multiple streams with batched inference (i.e. batch-size 8 for 8 streams)
# source = 'path/to/list.streams'  # *.streams text file with one streaming address per row

# Run inference on the source
result = model(source=0, show=True, conf=0.6 ,save=True)
# results = model(source, stream=True)  # generator of Results objects