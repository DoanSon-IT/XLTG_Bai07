import os
import cv2
import numpy as np

def load_images_and_labels(folder_path):
    images = []
    labels = []
    for label_folder in ["Chicken", "Duck"]:
        image_path = os.path.join(folder_path, label_folder)
        label_path = os.path.join(image_path, "Label")

        for filename in os.listdir(image_path):
            img_path = os.path.join(image_path, filename)
            if os.path.isfile(img_path) and not img_path.endswith("Label"):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (255, 255))
                    images.append(img)
                    labels.append(label_folder) 

    return np.array(images), np.array(labels)
