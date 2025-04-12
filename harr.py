import cv2
#from google.colab.patches import cv2_imshow
import numpy as np
import streamlit as st
import tempfile
import time

IMAGE_SIZE = (64, 64)  # All images will be resized to 64x64 for HOG
LOW_THRESHOLD = 3  # Less than 3 detections -> Low Traffic
HIGH_THRESHOLD = 6  # 3-5 detections -> Medium Traffic; 6 or more -> High Traffic

# Load the Haar cascade
vehicle_cascade = cv2.CascadeClassifier("haarcascade_car.xml")
if vehicle_cascade.empty():
    print("Error: Cascade file not loaded")
    exit()

def harrmain():

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
            image_proc, scale_factor = hog.preprocess_image_for_detection(image, max_dim=360)

            # Use sliding-window detection on single image
            st.info(
                "Detecting vehicles in single image using sliding window approach. Please wait..."
            )
            boxes = haarcascade_car_detect(image_proc, model, device, threshold=0.6)

            vehicle_count = len(boxes)
            # Draw bounding boxes on the image
            for x1, y1, x2, y2 in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

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
            stframe = st.empty()
            density_info = st.empty()

            if not cap.isOpened():
                print("Error: Unable to open video")
                exit()

            # Region of interest (ROI) - adjust based on your video
            roi_y_start = 150  # Skip top portion (sky/background)
            roi_y_end = 600    # Bottom of frame

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize and select ROI
                frame = cv2.resize(frame, (800, 600))
                roi = frame[roi_y_start:roi_y_end, :]

                # Convert to grayscale and enhance contrast
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Adaptive thresholding to highlight vehicles
                thresh = cv2.adaptiveThreshold(gray, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 2)

                # Morphological operations
                kernel = np.ones((3,3), np.uint8)
                processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # Detect vehicles with optimized parameters
                vehicles = vehicle_cascade.detectMultiScale(
                    processed,
                    scaleFactor=1.05,  # More sensitive to size variations
                    minNeighbors=2,    # Lower for more detections
                    minSize=(40, 40),  # Minimum vehicle size
                    maxSize=(300, 300) # Maximum vehicle size
                )

                # Draw bounding boxes
                vehicle_count = 0
                for (x, y, w, h) in vehicles:
                    vehicle_count = vehicle_count + 1
                    # Convert ROI coordinates back to full frame
                    abs_y = y + roi_y_start
                    cv2.rectangle(frame, (x, abs_y), (x+w, abs_y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'Car {w}x{h}', (x, abs_y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)     
           
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

                time.sleep(0.03)
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":    
    harrmain()