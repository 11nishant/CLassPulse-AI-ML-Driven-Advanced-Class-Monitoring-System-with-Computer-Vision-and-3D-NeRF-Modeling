import os
import cv2
import face_recognition
import numpy as np

# Define paths
images_dir = "~/Downloads/test_hostel_data/Images"
processed_dir = "~/Downloads/test_hostel_data/ProcessedImages"

# Expand '~' in paths
images_dir = os.path.expanduser(images_dir)
processed_dir = os.path.expanduser(processed_dir)

# Create 'ProcessedImages' directory if it doesn't exist
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# Process each sub-directory (p001, p002, ...)
for person in os.listdir(images_dir):
    person_dir = os.path.join(images_dir, person)
    
    if os.path.isdir(person_dir):
        # Create corresponding directory in 'ProcessedImages'
        processed_person_dir = os.path.join(processed_dir, person)
        os.makedirs(processed_person_dir, exist_ok=True)

        # Process each image
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {image_path}, skipping.")
                continue

            # Detect face
            face_locations = face_recognition.face_locations(image)
            if face_locations:
                # Get the first face location (top, right, bottom, left)
                top, right, bottom, left = face_locations[0]

                # Adjust the box to be larger (50% more around the face)
                height, width, _ = image.shape
                padding = int((bottom - top) * 0.5)  # 50% padding for complete face and some noise

                # Ensure padding doesn't exceed image dimensions
                top = max(0, top - padding)
                bottom = min(height, bottom + padding)
                left = max(0, left - padding)
                right = min(width, right + padding)

                # Crop the face (with padding)
                face_image = image[top:bottom, left:right]

                # Create a mask with the same dimensions as the cropped face image
                mask = np.zeros(face_image.shape[:2], dtype=np.uint8)

                # Draw the filled rectangle (white for the face area)
                mask = cv2.rectangle(mask, (0, 0), (right-left, bottom-top), 255, -1)

                # Create a 4-channel (RGBA) version of the face image
                face_with_alpha = cv2.cvtColor(face_image, cv2.COLOR_BGR2BGRA)

                # Apply the mask to the alpha channel
                face_with_alpha[:, :, 3] = mask

                # Ensure the filename only appends "_box" once
                base_filename, ext = os.path.splitext(image_file)
                processed_image_path = os.path.join(processed_person_dir, f"{base_filename}_box.png")

                # Save the processed face image without reusing names
                if not os.path.exists(processed_image_path):
                    cv2.imwrite(processed_image_path, face_with_alpha)
                    print(f"Processed and saved face for {image_file}")
                else:
                    print(f"File {processed_image_path} already exists, skipping.")

            else:
                print(f"No face detected in {image_file}, skipping.")

print("All faces processed and saved successfully!")

