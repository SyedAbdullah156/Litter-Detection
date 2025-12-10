"""
Making Inference Script
Uses: Alredy trained YOLO model for prediction
"""
import os
from ultralytics import YOLO
import torch

class LitterDetector:
    """Production inference pipeline"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
    def detect_image(self, image_path):
        """Detect litter in a single image"""
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=True,
            show_labels=True,
            show_conf=True
        )
        return results
    
    def detect_video(self, video_path, output_path='output_video.mp4'):
        """Detect litter in video"""
        results = self.model.predict(
            source=video_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=True,
            show_labels=True,
            show_conf=True,
            stream=True
        )
        
        # Process results
        for r in results:
            pass  # Results are automatically saved
        
        print(f"✅ Video processed and saved")
        return results
    
    def detect_webcam(self):
        """Real-time detection from webcam"""
        results = self.model.predict(
            source=0,  # 0 for default webcam
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            show=True,
            stream=True
        )
        
        for r in results:
            pass

if __name__ == "__main__":

    # --- 1. CONFIGURATION ---
    # Update this path to point to your trained model
    MODEL_WEIGHTS = "/home/luc/ai-ml/MLOPS/PROJECT/runs/yolov8s_modified8/weights/best.pt"
    
    # Update these paths to files you want to test
    TEST_IMAGE = "/home/luc/ai-ml/MLOPS/PROJECT/data/taco_dataset_processed_modified/test/images/000022_JPG_jpg.rf.99a53a58918b5a5552244ad533a6b296.jpg"
    
    TEST_VIDEO = "/home/luc/ai-ml/MLOPS/PROJECT/data/test_videos/vecteezy_river-garbage-pollution_2015570.mp4"

    # Verify model exists
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found at: {MODEL_WEIGHTS}")
        print("Please update 'MODEL_WEIGHTS' path in the script.")
        exit()

    # --- 2. INITIALIZE DETECTOR ---
    print(f"Loading Litter Detector with model: {MODEL_WEIGHTS}...")
    detector = LitterDetector(
        model_path=MODEL_WEIGHTS,
        conf_threshold=0.30,  # Ignore weak detections (<30%)
        iou_threshold=0.45    # NMS threshold
    )

    # --- 3. RUN INFERENCE (Uncomment the mode you want) ---

    # # # === MODE A: Single Image ===
    # if os.path.exists(TEST_IMAGE):
    #     print(f"Running inference on image: {TEST_IMAGE}")
    #     results = detector.detect_image(TEST_IMAGE)
    #     print(f"Image saved to: {results[0].save_dir}")
    # else:
    #     print(f"Warning: Test image '{TEST_IMAGE}' not found.")

    # === MODE B: Video File ===
    if os.path.exists(TEST_VIDEO):
        print(f"🎥 Running inference on video: {TEST_VIDEO}")
        detector.detect_video(TEST_VIDEO)
    else:
        print(f"Warning: Test video '{TEST_VIDEO}' not found.")

    # === MODE C: Live Webcam ===
    # print("Starting Webcam Inference... Press 'q' in the window to quit.")
    # detector.detect_webcam()