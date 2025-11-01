from ultralytics import YOLO
import cv2
import numpy as np
from utils.visualization import draw_predictions
from pathlib import Path
import time

def load_model(weights_path: str = "yolov8n.pt"):
    """
    Load a YOLO model
    Args:
        weights_path: Path to the model weights file
    Returns:
        YOLO model instance
    """
    return YOLO(weights_path)

def process_video(model, source, conf_threshold: float = 0.25, save_output: bool = False):
    """
    Process video stream (file or webcam) for object detection
    Args:
        model: YOLO model instance
        source: Video file path or camera index (0 for webcam)
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output video
    """
    # Open video capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        output_path = 'output_webcam.mp4'
    else:
        cap = cv2.VideoCapture(source)
        output_path = f'output_{Path(source).stem}.mp4'
    
    if not cap.isOpened():
        print("Error: Couldn't open video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if saving is enabled
    out = None
    if save_output:
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )

    # Process video frames
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(frame, conf=conf_threshold)[0]
        
        # Draw predictions
        annotated_frame = draw_predictions(frame, results)

        # Calculate and display FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time
        cv2.putText(annotated_frame, f'FPS: {fps_current:.1f}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Object Detection', annotated_frame)

        # Save frame if enabled
        if save_output and out is not None:
            out.write(annotated_frame)

        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

def main():
    # Initialize model (will download pretrained weights if not available)
    model = load_model()
    
    # Choose video source
    # For webcam, use:
    # source = 0
    # For video file, use:
    source = "data/sample.mp4"  # Replace with your video file path
    
    if not isinstance(source, int) and not Path(source).exists():
        print(f"Error: Video file not found at {source}")
        return
    
    # Process video with detection
    process_video(model, source, conf_threshold=0.25, save_output=True)

if __name__ == "__main__":
    main()
