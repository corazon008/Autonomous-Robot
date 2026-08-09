from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates 'yolo26n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo26n_ncnn_model")

# Run inference
results = ncnn_model("https://ultralytics.com/images/bus.jpg")
