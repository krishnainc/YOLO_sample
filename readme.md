
1. **Prepare Your Videos**
   ```
   data/
   └── videos/
       ├── video1.mp4    # Put your videos here
       ├── video2.mp4
       └── ...
   ```
   - Place all your videos that contain objects you want to detect in the `data/videos` folder

2. **Extract Frames from Videos**
   ```bash
   python utils/video_to_frames.py
   ```
   - This script will:
     - Extract frames from your videos
     - Create training (80%) and validation (20%) splits
     - Save frames in `dataset/train/images` and `dataset/val/images`

3. **Label Your Data**
   - Install a labeling tool like LabelImg:
     ```bash
     pip install labelImg
     ```
   - Start labeling:
     ```bash
     labelImg dataset/train/images
     ```
   - For each image:
     - Draw boxes around objects you want to detect
     - Assign class names (e.g., "car", "person", "dog")
     - Save in YOLO format
   - Repeat for validation images in `dataset/val/images`
   
4. **Configure Your Dataset**
   - Edit data.yaml:
     ```yaml
     path: dataset
     train: train/images
     val: val/images
     
     names:
       0: your_class1    # e.g., car
       1: your_class2    # e.g., person
       # Add all your classes
     ```

5. **Start Training**
   - Open train.py and adjust training parameters if needed:
     ```python
     config = {
         "data_yaml": "dataset/data.yaml",
         "model_type": "yolov8n.pt",  # n=nano, s=small, m=medium, l=large
         "epochs": 100,
         "imgsz": 640,
         "batch_size": 16,
         "device": "cpu"  # use "0" if you have a GPU
     }
     ```
   - Start training:
     ```bash
     python train.py
     ```
   - Training will:
     - Download pre-trained weights
     - Train on your custom dataset
     - Save the best model in `runs/train/yolov8_custom`

6. **Test Your Model**
   - After training, update app.py to use your trained model:
     ```python
     # In app.py, change the model path:
     model = load_model("runs/train/yolov8_custom/weights/best.pt")
     ```
   - Test with a video:
     ```bash
     python app.py
     ```



