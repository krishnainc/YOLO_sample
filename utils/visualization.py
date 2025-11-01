import cv2
import numpy as np

def draw_predictions(image, results):
    """
    Draw bounding boxes and labels on the image
    Args:
        image: Original image
        results: YOLO model prediction results
    Returns:
        Annotated image
    """
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    
    # Get the boxes and class IDs
    boxes = results.boxes.data.tolist()
    
    # Draw each detection
    for box in boxes:
        x1, y1, x2, y2, score, class_id = box
        
        # Convert coordinates to integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Get class name
        class_name = results.names[int(class_id)]
        
        # Draw bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"{class_name} {score:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_image, (x1, y1 - label_height - 10), (x1 + label_width, y1), (0, 255, 0), -1)
        cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated_image