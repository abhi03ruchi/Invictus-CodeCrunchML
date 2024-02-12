import os
import cv2
import numpy as np
from skimage import exposure
from skimage.feature import hog
import dlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Function for pre-processing the image
def preprocess_image(image):
    # Crop the image to remove unwanted regions
    # Assuming you know the region to crop, adjust accordingly
    cropped_image = image[y1:y2, x1:x2]

    # Resize the image to the desired input size of CNN
    resized_image = cv2.resize(cropped_image, (224, 224))

    # Apply intensity normalization using MinMax scaling
    normalized_image = exposure.rescale_intensity(resized_image, in_range='image', out_range=(0, 255))

    # Convert to float and normalize to range [0, 1]
    normalized_image = normalized_image.astype("float") / 255.0

    return normalized_image

# Function for extracting HOG features
def extract_hog_features(image):
    # Use skimage to compute HOG features
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return hog_features.flatten()

# Function for extracting facial landmarks
def extract_facial_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Convert image to grayscale for facial landmark detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) > 0:
        # Assume only one face in the image
        face = faces[0]

        # Get the facial landmarks
        landmarks = predictor(gray, face)

        # Select subset of landmarks around mouth and eye
        selected_landmarks = np.array([landmarks.part(i).x for i in range(48, 68)] +
                                      [landmarks.part(i).x for i in range(36, 42)])

        return selected_landmarks

    # Return None if no face is detected
    return None

# Load and preprocess the dataset
def load_dataset(base_path):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    images = []
    labels = []

    for emotion in emotions:
        for dataset_type in ["train", "test"]:
            folder_path = os.path.join(base_path, dataset_type, emotion)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg"):
                    file_path = os.path.join(folder_path, filename)
                    image = cv2.imread(file_path)
                   
                    # Preprocess the image
                    preprocessed_image = preprocess_image(image)
                   
                    # Extract HOG features and facial landmarks
                    hog_feature = extract_hog_features(preprocessed_image)
                    landmarks = extract_facial_landmarks(preprocessed_image)

                    if landmarks is not None:
                        combined_feature = np.concatenate([hog_feature, landmarks])
                        images.append(combined_feature)
                        labels.append(emotions.index(emotion))

    return np.array(images), np.array(labels)
