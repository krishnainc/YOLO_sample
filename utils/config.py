# YOLOv8 Configuration

# Model Configuration
MODEL_CONFIG = {
    'weights': 'yolov8n.pt',  # Path to model weights (will download if not present)
    'conf_threshold': 0.25,    # Confidence threshold for detections
    'iou_threshold': 0.45,     # NMS IOU threshold
}

# Inference Configuration
INFERENCE_CONFIG = {
    'device': 'cpu',          # Device to run inference on ('cpu' or 'cuda' for GPU)
    'half': False,            # Use FP16 half-precision inference
    'classes': None,          # Filter by class (None = all classes)
}

# Visualization Configuration
VIS_CONFIG = {
    'box_color': (0, 255, 0),  # BGR color for bounding boxes
    'text_color': (0, 0, 0),   # BGR color for text
    'line_thickness': 2,        # Thickness of bounding box lines
    'font_scale': 0.5,         # Font scale for class labels
    'font': 0,                 # Font type (0 = cv2.FONT_HERSHEY_SIMPLEX)
}