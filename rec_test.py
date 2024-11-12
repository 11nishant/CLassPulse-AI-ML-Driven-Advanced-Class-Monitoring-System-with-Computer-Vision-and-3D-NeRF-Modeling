import cv2
import numpy as np
from mtcnn import MTCNN
import inceptionresnet
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load the InceptionResNet model
model = inceptionresnet.InceptionResNetV2()

# Load the face encodings
with open("hostel_face_encodings.pkl", "rb") as f:
    face_encodings_dict = pickle.load(f)

# Create a reverse lookup for person names
known_face_names = list(face_encodings_dict.keys())
known_face_encodings = [np.mean(encodings, axis=0) for encodings in face_encodings_dict.values()]

# Initialize MTCNN detector
detector = MTCNN()

# Path to 'RawImages'
raw_images_dir = os.path.expanduser("~/Downloads/test_hostel_data/RawImages")

# Create a directory for recognized images
recognized_dir = os.path.join(raw_images_dir, "RecognizedImages")
if not os.path.exists(recognized_dir):
    os.makedirs(recognized_dir)

def preprocess_face(img, required_size=(160, 160)):
    img = cv2.resize(img, required_size)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = (img - 127.5) / 128.0
    return img

def get_embedding(face):
    face = preprocess_face(face)
    embedding = model.predict(face)
    return embedding[0]

def recognize_face(embedding, threshold=0.7):
    similarities = cosine_similarity([embedding], known_face_encodings)[0]
    best_match_index = np.argmax(similarities)
    
    if similarities[best_match_index] >= threshold:
        return known_face_names[best_match_index], similarities[best_match_index]
    return "Unknown", similarities[best_match_index]

def process_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    faces = detector.detect_faces(image_rgb)
    
    for face in faces:
        x, y, w, h = face['box']
        face_img = image_rgb[y:y+h, x:x+w]
        
        # Get face embedding
        embedding = get_embedding(face_img)
        
        # Recognize face
        name, similarity = recognize_face(embedding)
        
        # Draw bounding box and label
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{name} ({similarity:.2f})"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    return image

# Process each image in the RawImages directory
for image_file in os.listdir(raw_images_dir):
    image_path = os.path.join(raw_images_dir, image_file)
    
    # Check if the path is a file and has a valid image extension
    if os.path.isfile(image_path) and image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            # Process the image
            recognized_image = process_image(image_path)
            
            # Save the recognized image to the RecognizedImages directory
            recognized_image_path = os.path.join(recognized_dir, f"recognized_{image_file}")
            cv2.imwrite(recognized_image_path, recognized_image)
            
            print(f"Processed and saved recognized image: {recognized_image_path}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
    else:
        print(f"Skipping non-image file or directory: {image_path}")

print("All images processed and saved successfully!")