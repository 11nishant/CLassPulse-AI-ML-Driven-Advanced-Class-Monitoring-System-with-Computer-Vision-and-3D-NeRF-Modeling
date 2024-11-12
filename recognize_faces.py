import os
import cv2
import face_recognition
import pickle
import numpy as np

# Load the face encodings
with open("hostel_face_encodings.pkl", "rb") as f:
    face_encodings_dict = pickle.load(f)

# Create a reverse lookup for person names
known_face_names = list(face_encodings_dict.keys())
known_face_encodings = [encoding[0] for encoding in face_encodings_dict.values()]

# Path to 'RawImages'
raw_images_dir = os.path.expanduser("~/Downloads/test_hostel_data/RawImages")

# Create a directory for recognized images
recognized_dir = os.path.join(raw_images_dir, "RecognizedImages")
if not os.path.exists(recognized_dir):
    os.makedirs(recognized_dir)

# Process each image in the RawImages directory
for image_file in os.listdir(raw_images_dir):
    image_path = os.path.join(raw_images_dir, image_file)

    # Check if the path is a file and has a valid image extension
    if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = face_recognition.load_image_file(image_path)

        # Find all face locations and encodings in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Convert the image to BGR format for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Loop through each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the detected face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a rectangle around the face and label it
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the recognized image to the RecognizedImages directory
        recognized_image_path = os.path.join(recognized_dir, image_file)
        cv2.imwrite(recognized_image_path, image_bgr)

        print(f"Processed and saved recognized image: {recognized_image_path}")
    else:
        print(f"Skipping non-image file or directory: {image_path}")

print("All images processed and saved successfully!")

