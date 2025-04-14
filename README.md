# Intelligent Traffic Monitoring System

# Approach 1

## HOG+SVM

This approach implements a vehicle detection and traffic density estimation system using a combination of Histogram of Oriented Gradients (HOG) for feature extraction and a Linear Support Vector Machine (SVM) for classification. The system is built as a Streamlit application that supports both static images (using a sliding-window detection approach) and video streams (using background subtraction for moving objects).

## Overview

The approach:
- **Trains a Vehicle Detector**: Uses a dataset of vehicle and non-vehicle images to train a Linear SVM with HOG features.
- **Detects Vehicles in Static Images**: Uses a multi-scale sliding-window approach on (possibly downscaled) images to detect vehicles.
- **Detects Vehicles in Videos**: Leverages background subtraction combined with the HOG+SVM pipeline to detect moving vehicles.
- **Estimates Traffic Density**: Based on the number of detected vehicles, the traffic density is classified as:
  - **Low Traffic**: Less than 3 detections.
  - **Medium Traffic**: Between 3 and 5 detections.
  - **High Traffic**: 6 or more detections.
- **Enhances Image Quality**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve detection performance under low contrast conditions.

## Features

- **HOG Feature Extraction** for robust vehicle representation.
- **SVM Classifier** for vehicle vs. non-vehicle detection.
- **Sliding Window Detection** for static images.
- **Background Subtraction** for video-based detection.
- **High-Resolution Image Preprocessing**: Downscales 4K/8K images to reduce computation and maps detection results back to original resolution.
- **Streamlit Interface**: Easy-to-use web interface for uploading images/videos and visualizing results.

## Prerequisites

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- OpenCV (`opencv-python`)
- NumPy
- scikit-image
- scikit-learn
- joblib

2.	Usage:
For Images: Upload a static image (JPG, JPEG, or PNG). The application will downscale high-resolution images if necessary, run a sliding-window detector, and display bounding boxes and traffic density.

For Videos: Upload a video file (MP4 or AVI). The app will process video frames using background subtraction and display real-time vehicle detection along with the traffic density classification.

3. Code Structure
  •	HOG Feature Extraction:
  Located in extract_hog_features() function – computes HOG descriptors for given images.
  •	Dataset Loading & Training:
  The load_dataset() and train_model() functions load image data, extract features, and train the Linear SVM. The model is saved using joblib.
  •	Detection Pipeline:
  •	Static Images: Uses detect_vehicles_in_image_sliding_window() for multi-scale sliding-window detection.
  •	Videos: Uses detect_vehicles_in_frame() combined with background subtraction.
  •	CLAHE Enhancement:
  Applied through the apply_clahe() function to boost contrast.
  •	Preprocessing:
  High-resolution images are downscaled in preprocess_image_for_detection().
  •	Streamlit Interface:
  The main application logic resides in the main() function which handles file uploads and result visualization.

4. Troubleshooting & Tips
	•	High-Resolution Images:
The app automatically downscales images larger than the specified max_dim (default is 360). Adjust the max_dim parameter in preprocess_image_for_detection() if needed.
	•	False Positives/Negatives:
	•	Tweak parameters such as step_size, scale_factors, and decision_threshold in detect_vehicles_in_image_sliding_window().
	•	Retrain the SVM with more diverse training data if detections are not satisfactory.
	•	Performance:
The sliding-window approach can be computationally intensive on high-resolution images. Consider running the app on a machine with sufficient processing power or further optimizing window parameters.

# Approach 2

## Haar Cascade Classifier

This approach uses a combination of Haar Cascade detection, image preprocessing, and morphological operations to effectively detect vehicles in video frames. The code is capable of processing real-time video, highlighting detected vehicles with bounding boxes, and displaying the results interactively. By tuning the parameters of the Haar Cascade detector and image processing techniques, this method can be adapted to different environments and videos for vehicle detection tasks.

## Overview

This approach utilizes Haar Cascade Classifier for vehicle detection in video frames. By processing video frames through grayscale conversion, adaptive thresholding, and morphological operations, it detects and highlights vehicles in real-time. It works by analyzing a defined Region of Interest (ROI) in the video/image to identify vehicles with the help of the pre-trained Haar Cascade model (haarcascade_car.xml).

## Features

- **Real-time Vehicle Detection** Detects vehicles frame-by-frame in video files.
- **ROI Selection** Focuses on specific areas of the frame, excluding irrelevant regions like the sky.
- **Image Preprocessing** Uses grayscale conversion, contrast enhancement, Gaussian blur, and adaptive thresholding for better detection.
- **Bounding Boxes** Draws bounding boxes around detected vehicles for visualization.

## Prerequisites

- Python 3.7 or higher
- OpenCV
- NumPy

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Aryank47/Intelligent-traffic-monitoring-system.git
   cd Intelligent-traffic-monitoring-system
   ```

2.	Create a Virtual Environment (Optional but Recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

3.	Install the Required Dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  If you don’t have a requirements.txt, you can install dependencies manually:
  ```bash
  pip install streamlit opencv-python joblib numpy scikit-image scikit-learn
  ```


Dataset Setup
	1.	Download Datasets:
	•	Vehicles Dataset: Download Vehicles Dataset
	•	Non-Vehicles Dataset: Download Non-Vehicles Dataset
	2.	Extract the Archives:

Create a directory structure as follows:
```
  data/
    ├── vehicles/       # Contains folders with vehicle images
    └── non-vehicles/   # Contains folders with non-vehicle images
```

Training the Model

The application can automatically train the model if the trained model files (vehicle_classifier.pkl and scaler.pkl) are not found. To manually trigger training, simply run the Streamlit app; the training process will run and save the model files.

Running the Application
	1.	Launch the App:
  ```bash
  streamlit run main.py
  ```
