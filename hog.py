import glob
import os
import tempfile
import time

import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import hog
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -------------------- Configuration --------------------
IMAGE_SIZE = (64, 64)  # All images will be resized to 64x64 for HOG
LOW_THRESHOLD = 3  # Less than 3 detections -> Low Traffic
HIGH_THRESHOLD = 6  # 3-5 detections -> Medium Traffic; 6 or more -> High Traffic

MODEL_PATH = "vehicle_classifier.pkl"
SCALER_PATH = "scaler.pkl"

DATASET_VEHICLE_DIR = os.path.join("data", "vehicles")
DATASET_NONVEHICLE_DIR = os.path.join("data", "non-vehicles")


# -------------------- HOG Feature Extraction --------------------
def extract_hog_features(img, visualize=False):
    """
    Given a grayscale image, compute HOG features.
    """
    if visualize:
        features, hog_image = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            visualize=True,
        )
        return features, hog_image
    else:
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            transform_sqrt=True,
            visualize=False,
        )
        return features


# -------------------- Dataset Loading --------------------
def load_dataset():
    """
    Loads vehicle and non-vehicle images, converts them to grayscale,
    resizes them, and extracts HOG features.
    """
    features = []
    labels = []

    # Process vehicle images (label 1)
    vehicle_files = (
        glob.glob(os.path.join(DATASET_VEHICLE_DIR, "*/*.png"))
        + glob.glob(os.path.join(DATASET_VEHICLE_DIR, "*/*.jpg"))
        + glob.glob(os.path.join(DATASET_VEHICLE_DIR, "*/*.jpeg"))
    )
    for file in vehicle_files:
        img = cv2.imread(file)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)
        hog_feat = extract_hog_features(resized)
        features.append(hog_feat)
        labels.append(1)

    # Process non-vehicle images (label 0)
    nonvehicle_files = (
        glob.glob(os.path.join(DATASET_NONVEHICLE_DIR, "*/*.png"))
        + glob.glob(os.path.join(DATASET_NONVEHICLE_DIR, "*/*.jpg"))
        + glob.glob(os.path.join(DATASET_NONVEHICLE_DIR, "*/*.jpeg"))
    )
    for file in nonvehicle_files:
        img = cv2.imread(file)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, IMAGE_SIZE)
        hog_feat = extract_hog_features(resized)
        features.append(hog_feat)
        labels.append(0)

    return np.array(features), np.array(labels)


# -------------------- Model Training --------------------
def train_model():
    st.write(
        "Training vehicle detection model using HOG+SVM. This may take several minutes..."
    )
    X, y = load_dataset()
    st.write(f"Loaded {len(X)} samples for training.")

    # Scale features
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # (Optional) Split data to evaluate accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    clf = LinearSVC(dual=True, max_iter=150000)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    st.write(f"Validation Accuracy: {acc * 100:.2f}%")

    # Calculate other metrics
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(
        f"Precision: {precision * 100:.2f}%, Recall: {recall * 100:.2f}%, F1-Score: {f1 * 100:.2f}%"
    )
    # Or use a full classification report:
    # report = classification_report(y_test, y_pred)
    # st.text("Classification Report:\n" + report)

    # Save the trained classifier and scaler for later use
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.write("Model training complete and saved.")
    return clf, scaler


def load_model():
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return clf, scaler


# ---------------------- CLAHE (Contrast Limited Adaptive Histogram Equalization) ----------------


def apply_clahe(frame):
    """
    Applies CLAHE to enhance the contrast of the input frame.
    Converts the image to LAB color space, applies CLAHE to the L-channel,
    and then converts back to RGB.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced


# -------------------- NMS (Non-Maximum Suppression) --------------------


def non_max_suppression(boxes, overlapThresh=0.4):
    """
    Basic NMS to merge overlapping bounding boxes.
    boxes: list of (x, y, w, h, score)
    overlapThresh: IOU threshold for overlap
    """
    if len(boxes) == 0:
        return []

    # Convert to float for accurate overlap computations
    boxes_np = np.array(boxes, dtype=np.float32)

    # Coordinates
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    w = boxes_np[:, 2]
    h = boxes_np[:, 3]
    scores = boxes_np[:, 4]

    x2 = x1 + w
    y2 = y1 + h

    # Sort by score (descending)
    idxs = np.argsort(scores)[::-1]

    pick = []
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # Compute width and height of overlap
        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)
        overlap = (w_overlap * h_overlap) / (
            (w[i] * h[i]) + (w[idxs[1:]] * h[idxs[1:]]) - (w_overlap * h_overlap)
        )

        # Delete all indexes where overlap > threshold
        idxs = idxs[1:][overlap <= overlapThresh]

    # Return picked boxes
    return boxes_np[pick].astype(np.int32)


# -------------------- Sliding-Window for Single Image --------------------
def preprocess_image_for_detection(image, max_dim=1080):
    """
    Downscale the image if its maximum dimension (width or height) is greater than max_dim.
    Returns the resized image and the scaling factor.
    """
    original_h, original_w = image.shape[:2]
    scale_factor = 1.0
    if max(original_w, original_h) > max_dim:
        scale_factor = max(original_w, original_h) / max_dim
        new_w = int(original_w / scale_factor)
        new_h = int(original_h / scale_factor)
        resized_image = cv2.resize(image, (new_w, new_h))
        return resized_image, scale_factor
    return image, scale_factor


def detect_vehicles_in_image_sliding_window(
    frame,
    clf,
    scaler,
    step_size=32,
    window_size=(64, 64),
    scale_factors=[1.0, 1.5, 2.0],
    decision_threshold=1.0,
):
    """
    Perform a multi-scale sliding window approach on a single image using HOG+SVM.
    Returns bounding boxes in the form (x, y, w, h, score).
    """
    boxes = []
    original_h, original_w = frame.shape[:2]

    # Convert frame to RGB (if not already) and grayscale
    frame_rgb = (
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
    )
    for scale in scale_factors:
        # Resize image
        new_w = int(original_w / scale)
        new_h = int(original_h / scale)
        if new_w < window_size[0] or new_h < window_size[1]:
            continue
        scaled_img = cv2.resize(frame_rgb, (new_w, new_h))
        gray = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2GRAY)

        # Slide window
        for y in range(0, new_h - window_size[1], step_size):
            for x in range(0, new_w - window_size[0], step_size):
                window = gray[y : y + window_size[1], x : x + window_size[0]]
                if window.shape[:2] != window_size:
                    continue
                # Extract HOG features
                hog_feat = extract_hog_features(window)
                hog_feat = hog_feat.reshape(1, -1)
                hog_feat_scaled = scaler.transform(hog_feat)

                # SVM decision function
                decision = clf.decision_function(hog_feat_scaled)[0]
                # If above threshold => likely a vehicle
                if decision > decision_threshold:
                    # Scale coordinates back to original
                    orig_x = int(x * scale)
                    orig_y = int(y * scale)
                    orig_w = int(window_size[0] * scale)
                    orig_h = int(window_size[1] * scale)
                    boxes.append((orig_x, orig_y, orig_w, orig_h, decision))
    # Apply Non-Maximum Suppression
    boxes_nms = non_max_suppression(boxes, overlapThresh=0.2)

    # Convert final bounding boxes to (x, y, w, h)
    final_boxes = [(b[0], b[1], b[2], b[3]) for b in boxes_nms]
    return final_boxes


# -------------------- Vehicle Detection on a Frame --------------------


def detect_vehicles_in_frame(frame, bg_subtractor, clf, scaler):
    """
    Uses CLAHE-enhanced frame for background subtraction to identify candidate regions,
    and then applies the trained SVM classifier (using HOG features) to verify vehicles.
    """
    # Enhance contrast using CLAHE
    enhanced_frame = apply_clahe(frame)

    # Apply background subtraction on the enhanced frame
    fg_mask = bg_subtractor.apply(enhanced_frame)

    # Optionally adjust morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours from the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Filter out small regions
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            # Resize ROI and convert to grayscale for HOG extraction
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            roi_resized = cv2.resize(roi_gray, IMAGE_SIZE)
            hog_feat = extract_hog_features(roi_resized)
            hog_feat = hog_feat.reshape(1, -1)
            hog_feat_scaled = scaler.transform(hog_feat)
            prediction = clf.predict(hog_feat_scaled)
            if prediction[0] == 1:
                vehicle_boxes.append((x, y, w, h))
    return vehicle_boxes, fg_mask


# -------------------- Main Streamlit Application --------------------
def hog_main():

    # Check if dataset directories exist
    if not os.path.exists(DATASET_VEHICLE_DIR) or not os.path.exists(
        DATASET_NONVEHICLE_DIR
    ):
        st.error(
            "Dataset not found. Please download the vehicle and non-vehicle images from the URLs below and extract them as follows:\n\n"
            "**Vehicles:** https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip\n\n"
            "**Non-Vehicles:** https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip\n\n"
            "Extract the archives so that the directory structure is:\n"
            "`data/vehicles` and `data/non-vehicles`"
        )
        return

    # Load or train the classifier model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        clf, scaler = train_model()
    else:
        clf, scaler = load_model()
        st.write("Loaded pre-trained model.")

    # File uploader for images/videos
    uploaded_file = st.file_uploader(
        "Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"]
    )

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()

        # -------------------- Image Processing --------------------
        if file_extension in ["jpg", "jpeg", "png"]:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format
            if image is None:
                st.error("Error reading the image.")
                return

            # Preprocess: Downscale if needed
            image_proc, scale_factor = preprocess_image_for_detection(
                image, max_dim=360
            )

            # Use sliding-window detection on single image
            st.info(
                "Detecting vehicles in single image using sliding window approach. Please wait..."
            )
            boxes = detect_vehicles_in_image_sliding_window(
                image_proc,
                clf,
                scaler,
                step_size=16,
                window_size=(64, 64),
                scale_factors=[1.0, 1.5, 2.0],
                decision_threshold=0.0,
            )

            # Adjust boxes to original image scale if downscaled earlier
            if not np.isclose(scale_factor, 1.0):
                boxes = [
                    (
                        int(x * scale_factor),
                        int(y * scale_factor),
                        int(w * scale_factor),
                        int(h * scale_factor),
                    )
                    for (x, y, w, h) in boxes
                ]

            vehicle_count = len(boxes)
            # Draw bounding boxes on the image
            for x, y, w, h in boxes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert back to RGB for display in Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(
                image_rgb,
                caption=f"Detected Vehicles: {vehicle_count}",
                use_container_width=True,
            )

            # Classify overall traffic density
            if vehicle_count < LOW_THRESHOLD:
                density = "Low Traffic"
            elif vehicle_count < HIGH_THRESHOLD:
                density = "Medium Traffic"
            else:
                density = "High Traffic"

            st.markdown(f"**Traffic Density:** {density}")

        # -------------------- Video Processing --------------------
        elif file_extension in ["mp4", "avi"]:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

            # Background subtractor for moving objects
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=50, detectShadows=True
            )
            stframe = st.empty()
            density_info = st.empty()

            frame_count = 0
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # NOTE: 'frame' is in BGR by default from OpenCV
                vehicle_boxes, fg_mask = detect_vehicles_in_frame(
                    frame, bg_subtractor, clf, scaler
                )
                vehicle_count = len(vehicle_boxes)

                # Draw bounding boxes
                for x, y, w, h in vehicle_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Classify traffic density
                if vehicle_count < LOW_THRESHOLD:
                    density = "Low Traffic"
                elif vehicle_count < HIGH_THRESHOLD:
                    density = "Medium Traffic"
                else:
                    density = "High Traffic"

                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(
                    frame_rgb,
                    caption=f"Traffic Density: {density} (Vehicles: {vehicle_count})",
                    use_container_width=True,
                )
                density_info.markdown(
                    f"**Detected Vehicles:** {vehicle_count} | **Traffic Density:** {density}"
                )
                frame_count += 1
                # Optionally, update FPS display every few frames:
                if frame_count % 10 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    effective_fps = frame_count / elapsed
                    st.write(f"Effective Processing FPS: {effective_fps:.2f}")

                # time.sleep(0.03)
            cap.release()
