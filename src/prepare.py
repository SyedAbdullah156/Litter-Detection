"""
Data Preparation Script
Uses: OpenCV for image processing
"""

import os
import cv2
import shutil   # Used for copying and pasting images from src to dest
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm # TO PRINT INTERACTIVE PROGRESS BAR
from pycocotools.coco import COCO # AS ANNOTATIONS FORMAT IS INITIALLY IN COCO

class TACODataProcessor:
    """Process TACO dataset from COCO format to YOLO format"""
    
    def __init__(self, taco_path, output_path, train_split=0.7, val_split=0.2):
        self.taco_path = Path(taco_path)
        self.output_path = Path(output_path)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1 - train_split - val_split
        
    def coco_to_yolo_bbox(self, bbox, img_width, img_height):
        """Convert COCO bbox [x, y, width, height] to YOLO [x_center, y_center, width, height] where in YOLO each coordinate is realtive to image_width and image_height (multiple b/w 0-1)."""
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return [x_center, y_center, width, height]
    
    def process_dataset(self):
        """Main processing pipeline"""
        print("Starting TACO dataset processing...")
        
        # Load COCO annotations
        ann_file = self.taco_path / 'annotations.json'
        print(ann_file)
        coco = COCO(str(ann_file))
        
        # Get all image ids (is a list [0 - 1499])
        img_ids = coco.getImgIds()
        
        # Split dataset into train and temp (val + test)
        train_ids, temp_ids = train_test_split(
            img_ids, 
            test_size=(1-self.train_split), 
            random_state=42
        )

        # Split dataset into val and test
        val_ids, test_ids = train_test_split(
            temp_ids, 
            test_size=(self.test_split/(self.test_split+self.val_split)), 
            random_state=42
        )
        
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
        # Process each split
        for split_name, ids in splits.items():
            self._process_split(coco, ids, split_name)
        
        print("Dataset processing completed!")
        
    def _process_split(self, coco, img_ids, split_name):
        """Process a single data split"""
        print(f"\nProcessing {split_name} split ({len(img_ids)} images)...")
        
        # Create directories
        img_dir = self.output_path / split_name / 'images'
        label_dir = self.output_path / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for img_id in tqdm(img_ids, desc=f"Processing {split_name}"):
            # Get image info
            img_info = coco.loadImgs(img_id)[0]
            img_width = img_info['width']
            img_height = img_info['height'] 
            img_filename = img_info['file_name']                                            # format: batch_1/000010.jpg
            
            # Copy and Validate image using OpenCV
            src_img = self.taco_path / img_filename                                         # format: ../data/taco_data/ + batch_1/000010.jpg
            dst_img = img_dir / img_filename.split("/")[1]                                  # format: ../data/taco_data_processed/{split_name}/images + 000010.jpg ; REMEMBER THAT HERE WE ARE REMOVING THE BATCHES IN OUTPUT DIRECTORY
            if src_img.exists():
                img = cv2.imread(str(src_img))

                if img is None:
                    raise ValueError(f"Cannot read image at {src_img}")

                if not cv2.imwrite(str(dst_img), img):
                    raise ValueError(f"Failed to write image to {dst_img}")
            
            # Get annotations for this image
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            # Convert to YOLO format
            yolo_annotations = []
            for ann in anns:
                if 'bbox' in ann:
                    category_id = ann['category_id']
                    bbox = ann['bbox']
                    yolo_bbox = self.coco_to_yolo_bbox(bbox, img_width, img_height)
                    yolo_ann = f"{category_id} {' '.join(map(str, yolo_bbox))}"             # format: 0 0.1 0.5 0.6 0.8
                    yolo_annotations.append(yolo_ann)
            
            # Save label file
            label_file = label_dir / f"{Path(img_filename.split('/')[1]).stem}.txt"         # format: 000010.jpg ; REMEMBER THAT HERE WE ARE REMOVING THE BATCHES
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))

if __name__ == "__main__":
    processor = TACODataProcessor(
        taco_path="../data/taco_data", 
        output_path="../data/taco_data_processed", 
        train_split=0.7, 
        val_split=0.2
    )
    processor.process_dataset()