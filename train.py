from ultralytics import YOLO

def train_model(
    data_yaml: str = "dataset/data.yaml",
    model_type: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "cpu"  # use "0" for GPU if available
):
    """
    Train a YOLO model on a custom dataset
    Args:
        data_yaml: Path to data configuration file
        model_type: Type of YOLO model to use
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        device: Device to train on ('cpu' or '0' for GPU)
    """
    # Load a model
    model = YOLO(model_type)  # load a pretrained model (recommended for training)

    # Train the model
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=50,  # early stopping patience
            save=True,  # save best checkpoint
            project="runs/train",  # project name
            name="yolov8_custom",  # experiment name
            exist_ok=True,  # overwrite existing experiment
            pretrained=True,  # start from pretrained weights
            optimizer="auto",  # optimizer (SGD, Adam, etc.)
            verbose=True,  # verbose output
            seed=42,  # random seed
            deterministic=True,  # deterministic training
        )
        print("Training completed successfully!")
        return results
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def main():
    # Configure training parameters
    config = {
        "data_yaml": "dataset/data.yaml",
        "model_type": "yolov8n.pt",  # can also use s, m, l, or x variants
        "epochs": 100,
        "imgsz": 640,
        "batch_size": 16,
        "device": "cpu"  # change to "0" if you have a GPU
    }
    
    # Start training
    print("Starting model training...")
    results = train_model(**config)
    
    if results:
        print(f"Model saved in: {results.save_dir}")

if __name__ == "__main__":
    main()