import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm


class_path = os.path.join("../data/dataset-resized", "cardboard")
images = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

train_imgs, temp_imgs = train_test_split(
    images, 
    test_size=(0.15 + 0.15),
    random_state=42
)

for i in tqdm(range(10000)):
    print(i)