from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26s.pt")


# Export the model to NCNN format
model.export(
    format="onnx",
    # imgsz=480,
    quantize="fp16",
    dynamic=True,
)  # creates 'yolo26n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo26n.onnx")

# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")
