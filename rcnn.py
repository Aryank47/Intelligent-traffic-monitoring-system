import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import torch
import numpy as np
import streamlit as st
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import cv2
import hog
import tempfile
import time


# -------------------- Configuration --------------------
IMAGE_SIZE = (64, 64)  # All images will be resized to 64x64 for HOG
LOW_THRESHOLD = 3  # Less than 3 detections -> Low Traffic
HIGH_THRESHOLD = 6  # 3-5 detections -> Medium Traffic; 6 or more -> High Traffic

MODEL_PATH = "RCNN_FineTune_epoch_18.pth"
# SCALER_PATH = "scaler.pkl"

TRAIN_DATASET_DIR = os.path.join("data", "vehicles")
TRAIN_LABEL_DIR = os.path.join("data", "vehicles")
TEST_DATASET_DIR = os.path.join("data", "non-vehicles")
TEST_LABEL_DIR = os.path.join("data", "vehicles")



class UADETRACDatasetTXT(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        annotation_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.txt'))

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        # Load annotations
        boxes = []
        labels = []
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    # Convert from normalized to absolute coordinates
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height
                    # Convert to (x_min, y_min, x_max, y_max)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # Assuming class_id starts from 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    customModel = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(customModel['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(customModel['optimizer_state_dict'])
    return model

def collate_fn(batch):
  return tuple(zip(*batch))

def train_model():
    transform = T.Compose([T.ToTensor()])
    dataset = UADETRACDatasetTXT(TRAIN_DATASET_DIR, TRAIN_LABEL_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=2)  # 1 class + background
    customModel = torch.load(f"{MODEL_PATH}RCNN_FineTune_epoch_9.pth")
    model.load_state_dict(customModel['model_state_dict'])
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(customModel['optimizer_state_dict'])

    # Resume from next epoch
    start_epoch = customModel['epoch'] + 2

    # Training loop
    num_epochs = 8
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, targets in tqdm(dataloader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        print(f"Epoch [{epoch+start_epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, f"RCNN_FineTune_epoch_{epoch+start_epoch}.pth")


def evaluate_image(image, model, device, threshold=0.5):
    
    # img = Image.open(image).convert("RGB")
    img_tensor = to_tensor(image).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    thresholdBoxes = []
    # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for box, score in zip(boxes, scores):
        if score > threshold:
            thresholdBoxes.append(box)
            # x1, y1, x2, y2 = box.astype(int)
            # cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    # plt.title("Detection Results")
    # plt.axis("off")
    # plt.show()
    return thresholdBoxes

def rcnnmain():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(2, device)
    model.eval()

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
            boxes = evaluate_image(image_proc, model, device, threshold=0.6)

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

            # Background subtractor for moving objects
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, varThreshold=50, detectShadows=True
            )
            stframe = st.empty()
            density_info = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # NOTE: 'frame' is in BGR by default from OpenCV
                vehicle_boxes = evaluate_image(frame, model, device, threshold=0.6) #detect_vehicles_in_frame(frame, bg_subtractor, clf, scaler)
                vehicle_count = len(vehicle_boxes)

                # Draw bounding boxes
                for x1, y1, x2, y2 in vehicle_boxes:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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


if __name__ == "__main__":
    evaluate_image("D:/Sem3/CV/Project/DETRAC_Upload/images/val/MVI_39031_img00002.jpg", threshold=0.8)