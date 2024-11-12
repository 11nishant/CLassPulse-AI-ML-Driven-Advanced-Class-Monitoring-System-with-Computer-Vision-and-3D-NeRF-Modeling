import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import dlib
dlib.DLIB_USE_CUDA = False

import pickle
import face_recognition

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path to 'ProcessedImages'
processed_images_dir = "~/Downloads/test_hostel_data/Images"
processed_images_dir = os.path.expanduser(processed_images_dir)

# Dictionary to hold encodings of each person
face_encodings_dict = {}

for person in os.listdir(processed_images_dir):
    person_dir = os.path.join(processed_images_dir, person)
    encodings = []
    
    # Loop through all processed images of the person
    for image_file in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_file)
        image = face_recognition.load_image_file(image_path)
        
        # Use HOG model for face detection (CPU-based)
        face_locations = face_recognition.face_locations(image, model="hog")
        
        # Get face encodings using CPU
        face_encodings = face_recognition.face_encodings(image, face_locations, model="small")
        
        # Ensure at least one face encoding is found
        if face_encodings:
            encodings.append(face_encodings[0])
    
    # Store multiple encodings for robustness
    if encodings:
        face_encodings_dict[person] = encodings

# Save encodings to a file
with open("hostel_face_encodings.pkl", "wb") as f:
    pickle.dump(face_encodings_dict, f)

print("Face encodings saved successfully!")
