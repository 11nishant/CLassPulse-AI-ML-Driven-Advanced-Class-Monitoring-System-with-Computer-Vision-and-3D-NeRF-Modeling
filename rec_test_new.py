# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:59:33 2022

@author: Shivam
"""

import os
import cv2
import dlib

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Define the path to the test_hostel_data directory
base_path = os.path.expanduser("~/Downloads/test_hostel_data")
images_dir = os.path.join(base_path, "RawImages")
processed_dir = os.path.join(base_path, "ProcessedImages")

# Create the ProcessedImages directory if it doesn't exist
os.makedirs(processed_dir, exist_ok=True)

# Loop through each subdirectory in the Images directory
for person_dir in os.listdir(images_dir):
    person_path = os.path.join(images_dir, person_dir)

    if os.path.isdir(person_path):  # Check if it is a directory
        for image_name in os.listdir(person_path):
            # Build the full image path
            image_path = os.path.join(person_path, image_name)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}. Skipping.")
                continue
            
            # Run the HOG face detector on the image data
            detected_faces = face_detector(image, 1)

            print("Found {} faces in the image file {}".format(len(detected_faces), image_path))

            # Create a sub-directory for the current image
            image_name_without_ext = os.path.splitext(image_name)[0]  # Get the image name without extension
            image_processed_dir = os.path.join(processed_dir, image_name_without_ext)
            os.makedirs(image_processed_dir, exist_ok=True)

            # Loop through each face we found in the image
            for i, face_rect in enumerate(detected_faces):
                # Detected faces are returned as an object with the coordinates
                print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(
                    i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

                # Crop the detected face
                alignedFace = image[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]

                # Save the aligned image to the specific sub-directory
                aligned_face_path = os.path.join(image_processed_dir, f"aligned_face_{i}.jpg")
                cv2.imwrite(aligned_face_path, alignedFace)
                print(f"Aligned face saved as {aligned_face_path}")

print("Face alignment completed.")
