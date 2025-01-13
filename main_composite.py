import os
import cv2
import csv
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
import joblib
from collections import defaultdict, Counter
from tensorflow.keras.models import load_model
import json


directory_path = "../Dataset/testset/testset/"

output_csv_name = "submission_composite.csv"

show_images = False







svm_model_path = "./svm_face_recognizer.pkl"

# Load the trained model
model = load_model("NN_model.keras")

# Load the label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)


# Register HEIF opener for Pillow
register_heif_opener()

# YuNet model path
yunuet_path = "./face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(
    model=yunuet_path,
    config="",
    input_size=(1280, 960),
    score_threshold=0.66,
    nms_threshold=0.3,
    top_k=500
)

# Load pretrained SVM model

svm_model = joblib.load(svm_model_path)

# Load MobileNetV2 for feature extraction
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)



def predict_label_nn(features, model, label_map):
    """
    Predicts the label and confidence for given features using the model.
    
    Parameters:
    - features: np.array, extracted features of the image (shape should be (1, num_features)).
    - model: Trained model loaded using keras.models.load_model.
    - label_map: Dictionary mapping class indices to label names.
    
    Returns:
    - predicted_label: str, the name of the predicted class.
    - confidence: float, the confidence value of the prediction.
    """
    # Ensure features are 2D (batch dimension)
    features = np.expand_dims(features, axis=0)  # Shape: (1, num_features)

    # Predict probabilities
    predictions = model.predict(features, verbose=0)  # Shape: (1, num_classes)
    predictions = predictions.flatten()  # Flatten to 1D

    # Get the predicted class index and confidence
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    # Get the class label
    reverse_label_map = {v: k for k, v in label_map.items()}
    predicted_label = reverse_label_map[predicted_class]

    return predicted_label, confidence


def extract_features(image):
    """
    Extract features from a preprocessed image using MobileNetV2.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image)  # Normalize image for MobileNetV2
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    features = feature_extractor.predict(image, verbose=0)  # Suppress progress bars
    return features.flatten()  # Flatten the features


def face_drawer(image, face,confidence, label="Unknown"):
    """
    Draw face bounding box, detection confidence, and a custom label on the image.
    """
    x, y, w, h = face[:4].astype(int)  # Face bounding box
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw detection confidence
    confidence_text = f"{confidence:.2f}"
    cv2.putText(image, confidence_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

    # Draw label (e.g., name)
    cv2.putText(image, label, (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

def add_filename_overlay(image, filename):
    """
    Add the filename as an overlay on the image.
    """
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 255, 0)  # Green color
    cv2.putText(overlay, filename, (10, 30), font, font_scale, color, thickness)
    return overlay


def detect_faces_in_frame(frame, face_predictions, frame_index, draw=True):
    """
    Detect faces, extract features, and predict names. Track predictions across frames.
    """
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)

    if faces is not None:
        # Sort faces by their x-coordinate (horizontal order)
        faces = sorted(faces, key=lambda face: face[0])  # Sort by x (horizontal position)

        for idx, face in enumerate(faces):
            x, y, w, h = face[:4].astype(int)

            # Ensure the bounding box is within the image dimensions
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)  # Ensure width is at least 1 pixel
            h = max(1, h)  # Ensure height is at least 1 pixel
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            # Crop the face and extract features
            cropped_face = frame[y:y2, x:x2]
            if cropped_face.size == 0:
                label = "Unknown"
            else:
                cropped_face = cv2.resize(cropped_face, (160, 160))
                features = extract_features(cropped_face)

                # Predict the name using the SVM model
                
                label_nn, confidence_nn = predict_label_nn(features, model, label_map)
                label_svm = svm_model.predict([features])[0]
                confidence_list_svm =  svm_model.predict_proba([features])
                confidence_svm = np.max(confidence_list_svm)
                
                confidence = max(confidence_nn, confidence_svm)
                if confidence == confidence_nn:
                    label = label_nn
                else:
                    label = label_svm

            # Track predictions by face index
            face_predictions[idx].append(label)

            # Draw the face with the predicted label
            if draw:
                face_drawer(frame, face, confidence_svm,label=label)


def load_file_for_opencv(file_path):
    """
    Load a file (HEIC, MP4, or JPG) and return a list of frames for OpenCV.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    frames = []

    if file_ext in [".heic", ".jpg", ".jpeg", ".png"]:  # Handle images
        try:
            if file_ext == ".heic":
                image = Image.open(file_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_array = np.array(image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                frames.append(image_array)  # Single image as one frame
            else:
                image_array = cv2.imread(file_path)
                if image_array is None:
                    raise ValueError("Failed to load image.")
                frames.append(image_array)  # Single image as one frame
        except Exception as e:
            print(f"Error loading image file: {e}")
    
    elif file_ext in [".mp4", ".avi", ".mkv"]:  # Handle videos
        try:
            video_capture = cv2.VideoCapture(file_path)
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                frames.append(frame)  # Add each frame to the list
            video_capture.release()
        except Exception as e:
            print(f"Error loading video file: {e}")
    
    else:
        print(f"Unsupported file format: {file_ext}")
    
    return frames


def resize_to_fit(image, max_width=800, max_height=600):
    """
    Resize the image or video frame to fit within the specified max dimensions.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def display_directory_contents_with_faces(directory, show_images=True, output_csv="predictions.csv"):
    """
    Display each image or video in the specified directory one by one with face detection and recognition.
    Print the most frequently predicted name for each face in videos, in horizontal order.
    """
    files = sorted(os.listdir(directory))  # Sort files alphabetically
    results = []  # Store results for CSV output

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            frames = load_file_for_opencv(file_path)
            base_name = os.path.splitext(file_name)[0]  # Get the filename without extension
            
            # Track predictions across frames
            face_predictions = defaultdict(list)

            for frame_index, frame in enumerate(frames):
                resized_frame = resize_to_fit(frame)
                detect_faces_in_frame(resized_frame, face_predictions, frame_index, draw=show_images)

                # Display each frame for videos if show_images is True
                if show_images:
                    frame_with_overlay = add_filename_overlay(resized_frame, f"{base_name}")
                    cv2.imshow("Media Viewer", frame_with_overlay)

                    if len(frames) > 1:  # Video
                        if cv2.waitKey(30) & 0xFF == ord('q'):  # Play at ~30 FPS
                            cv2.destroyAllWindows()
                            return
                    else:  # Single image
                        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Wait for 1 second
                            cv2.destroyAllWindows()
                            return

            # Calculate the most frequent name for each face in horizontal order
            final_predictions = [Counter(names).most_common(1)[0][0] for idx, names in sorted(face_predictions.items())]

            # Format results for printing and CSV
            if final_predictions:
                formatted_result = f"{base_name}: {'; '.join(final_predictions)}"
            else:
                formatted_result = f"{base_name}: nothing"

            print(formatted_result)
            results.append((str(int(base_name)), ";".join(final_predictions) if final_predictions else "nothing"))

    # Save results to CSV
    with open(output_csv, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image", "label_name"])
        csvwriter.writerows(results)

    cv2.destroyAllWindows()


# Directory containing images/videos


# Run the program with options
display_directory_contents_with_faces(directory_path, show_images=show_images, output_csv=output_csv_name)
