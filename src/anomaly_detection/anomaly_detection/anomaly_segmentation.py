import os
import cv2
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- Load model once ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Preprocessing
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2(),
])

# Model
MODEL_PATH = "/home/baurov/ros2_ws/src/anomaly_detection/models/AnomalySegment.pth"
model = smp.Unet(
    encoder_name="mobilenet_v2",
    in_channels=3,
    classes=1,
    activation=None   # sigmoid manually later
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def show_mask(pred_mask, win_name="Mask", max_size=800):
    """
    Show a binary mask scaled to fit on screen.
    """
    # scale mask to [0,255]
    mask_vis = (pred_mask * 255).astype(np.uint8)

    h, w = mask_vis.shape
    scale = min(max_size / h, max_size / w, 1.0)  # shrink only if too big
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(mask_vis, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized = cv2.resize(resized, (640, 480), interpolation=cv2.INTER_NEAREST)

    cv2.imshow(win_name, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segment_anomaly(img, threshold=0.5, save_path=None, vis=False):
    """
    Run segmentation on one image.

    Args:
        image_path (str): Path to input image
        threshold (float): Probability threshold for mask binarization
        save_path (str): If provided, save mask to this path

    Returns:
        np.ndarray: Binary mask (H, W) with values {0,1}
    """
    # --- Read & preprocess ---
    # img = cv2.imread(image_path)
    original_shape = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img)
    tensor_img = augmented["image"].unsqueeze(0).to(device)

    # --- Forward pass ---
    with torch.no_grad():
        pred = model(tensor_img)
        pred = torch.sigmoid(pred)  # apply sigmoid
        pred_mask = (pred[0, 0].cpu().numpy() > threshold).astype(np.uint8)


    
    pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    # --- Save if requested ---
    if save_path:
        mask_img = Image.fromarray(pred_mask * 255).convert("L")
        mask_img.save(save_path)

    if vis:
        # scale to half size for viewing
        show_mask(pred_mask)

    return pred_mask


# Example usage:
if __name__ == "__main__":
    img_path = "templates/altigen.jpg"
    img = cv2.imread(img_path)
    mask = segment_anomaly(img,vis=True)

    print("Mask shape:", mask.shape)
    
