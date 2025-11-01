import cv2
import os
from pathlib import Path
from tqdm import tqdm

def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 30,  # Extract every Nth frame
    min_frames: int = 100  # Minimum number of frames to extract
):
    """
    Extract frames from a video file
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame
        min_frames: Minimum number of frames to extract
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Adjust frame_interval if needed to meet min_frames requirement
    if total_frames // frame_interval < min_frames and total_frames >= min_frames:
        frame_interval = total_frames // min_frames
        print(f"Adjusted frame interval to {frame_interval} to get enough frames")
    
    # Initialize frame counter
    count = 0
    saved_count = 0
    
    # Create progress bar
    pbar = tqdm(total=total_frames, desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at specified intervals
        if count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    print(f"Extracted {saved_count} frames from {video_path}")
    return saved_count

def process_videos_for_training(
    video_dir: str,
    output_base_dir: str = "dataset",
    train_ratio: float = 0.8,
    frame_interval: int = 30
):
    """
    Process multiple videos and split frames into training and validation sets
    Args:
        video_dir: Directory containing video files
        output_base_dir: Base directory for dataset
        train_ratio: Ratio of frames to use for training
        frame_interval: Extract every Nth frame
    """
    # Create output directories
    train_dir = os.path.join(output_base_dir, "train/images")
    val_dir = os.path.join(output_base_dir, "val/images")
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing {video_file}...")
        
        # Create temporary directory for frames
        temp_dir = os.path.join(output_base_dir, "temp_frames")
        Path(temp_dir).mkdir(exist_ok=True)
        
        # Extract frames
        num_frames = extract_frames(video_path, temp_dir, frame_interval)
        
        if num_frames == 0:
            continue
            
        # Split frames into train and val sets
        frame_files = sorted(os.listdir(temp_dir))
        split_idx = int(len(frame_files) * train_ratio)
        
        # Move frames to train and val directories
        print("Splitting frames into train and validation sets...")
        for i, frame_file in enumerate(frame_files):
            source = os.path.join(temp_dir, frame_file)
            if i < split_idx:
                dest = os.path.join(train_dir, f"{Path(video_file).stem}_{frame_file}")
            else:
                dest = os.path.join(val_dir, f"{Path(video_file).stem}_{frame_file}")
            os.rename(source, dest)
        
        # Clean up temporary directory
        os.rmdir(temp_dir)
    
    # Print summary
    train_count = len(os.listdir(train_dir))
    val_count = len(os.listdir(val_dir))
    print(f"\nDataset creation complete!")
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")

def main():
    # Configuration
    config = {
        "video_dir": "data/videos",  # Put your videos here
        "output_base_dir": "dataset",
        "train_ratio": 0.8,  # 80% training, 20% validation
        "frame_interval": 30  # Extract every 30th frame
    }
    
    # Create videos directory if it doesn't exist
    Path(config["video_dir"]).mkdir(parents=True, exist_ok=True)
    
    print("Please put your video files in the 'data/videos' directory")
    print("Press Enter when ready to process videos...")
    input()
    
    # Process videos
    process_videos_for_training(**config)
    
    print("\nNext steps:")
    print("1. Label the extracted frames using a tool like LabelImg")
    print("2. Create label files in the train/labels and val/labels directories")
    print("3. Update the dataset/data.yaml file with your classes")
    print("4. Run train.py to start training")

if __name__ == "__main__":
    main()