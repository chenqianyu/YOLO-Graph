# coding=utf-8

# Standard Library Imports
import os
import re
import time

# Third Party Imports
import numpy as np
import cv2
from ultralytics import YOLO

# Local Application/Library Specific Imports
from base.settings import LoggerMixin


class ObjectDetection(LoggerMixin):


    def __init__(self, mapping=None):
        super().__init__()
        
        if mapping is None:
            self.mapping = {0: 'RBC', 1: 'WBC', 2: 'PLT'} 
        else:
            self.mapping = mapping


    def normalize_images(self, batch_imgs):
        """
        Normalizes a batch of single-channel images to three-channel images with pixel values scaled between 0 and 255.

        This function takes a list of single-channel images (grayscale) and processes each image to convert it into a 
        three-channel image (BGR format). It does this by stacking the single-channel image across the three channels. 
        Then, it normalizes the pixel values of these images to the range [0, 255] using OpenCV's `cv2.normalize` method. 
        The normalization is performed using the min-max normalization technique.

        Parameters:
        batch_imgs (list of numpy.ndarray): A list of images where each image is represented as a single-channel numpy 
                                            array (grayscale image).

        Returns:
        list of numpy.ndarray: A list of normalized images, where each image is a three-channel BGR image with pixel values 
                            scaled between 0 and 255.
        """
        normalized_images = []
        for img in batch_imgs:
            new_phase_image = np.dstack([img, img, img])
            new_phase_image = new_phase_image[:, :, :3]
            normalized_img = cv2.normalize(new_phase_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            normalized_images.append(normalized_img)
        return normalized_images


    def extract_index(self, filename):
        """
        Extracts the numeric index from a filename.
        
        Parameter:
            filename: The string representing the name of a file.
            
        Return:
            The extracted numeric index as an integer, or -1 if no index is found.
        """
        match = re.search(r'\d+\.png$', filename)
        if match:
            return int(match.group().split(".")[0])
        return -1  # Return a default value if no match is found


    def predict(self, img_path, weight_path, reconstructed_images, device):
        """
        Predicts on images using a YOLO model with given weights. Supports single images or image directories.

        Parameters:
            img_path: Path to an image or a directory of images.
            weight_path: Path to the model weights.
            reconstructed_images: List of pre-loaded images, used if non-empty.

        Return:
            A tuple (all_images, all_results, image_ids) with processed images, prediction results, and image IDs.
        """
        all_results, all_images, image_ids = [], [], []
        batch_size = self.config["detection_batch_size"]
        model = YOLO(weight_path)

        def process_batch(batch_imgs, batch_img_ids):
            """ Process a batch of images and update results and images lists. """
            all_images.extend(batch_imgs)
            image_ids.extend(batch_img_ids)
            predict_start_time = time.perf_counter()
            results = model(batch_imgs, imgsz=(self.config["img_height"], self.config["img_width"]), verbose=False, device=device)
            all_results.extend(results)
            return time.perf_counter() - predict_start_time

        total_predict_time = 0
        img_files = []
        if reconstructed_images:
            for i in range(0, len(reconstructed_images), batch_size):
                batch_imgs = reconstructed_images[i:i + batch_size]
                normalized_batch_imgs = self.normalize_images(batch_imgs)
                batch_img_ids = [f"{i+j}" for j in range(len(normalized_batch_imgs))]
                total_predict_time += process_batch(normalized_batch_imgs, batch_img_ids)
        else:
            img_files = [img_path] if img_path.endswith('.png') else sorted([f for f in os.listdir(img_path) if f.endswith('.png')], key=self.extract_index)
            for i in range(0, len(img_files), batch_size):
                batch_img_files = img_files[i:i + batch_size]
                batch_imgs = [cv2.imread(os.path.join(img_path, img_file)) for img_file in batch_img_files]
                total_predict_time += process_batch(batch_imgs, batch_img_files)

        if len(img_files) > 0:
            avg_predict_time = (total_predict_time / len(img_files)) * 1000
            self.logger.info(f'Average prediction time: {avg_predict_time} milliseconds')

        return all_images, all_results, image_ids
