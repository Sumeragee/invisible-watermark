import os
import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from imwatermark.rivaGan import RivaWatermark  # Needed for manual model load

input_file = 'kafka run back.jpg'
output_file = 'watermarked_image.png'
watermark_text = 'SH13'  # Must be exactly 4 characters for rivaGan
algorithm = 'rivaGan'    # Or switch to 'dwtDct' for longer text

# === Check if input file exists ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Input image not found: {input_file}")

# === Load the image ===
bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

# === Initialize the encoder ===
print("[ℹ️] Initializing encoder...")
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', watermark_text.encode('utf-8'))

# === Load rivaGan model manually if used ===
if algorithm == 'rivaGan':
    print("[ℹ️] Loading rivaGan model...")
    encoder._wm_encoder = RivaWatermark(encoder._watermarks, encoder._wmLen)
    encoder._wm_encoder.loadModel()

# === Encode watermark ===
print("[ℹ️] Encoding in progress...")
bgr_encoded = encoder.encode(bgr, algorithm)

if bgr_encoded is None:
    raise RuntimeError(f"[✘] Failed to encode watermark using algorithm '{algorithm}'")

# === Debug: Check encoded image properties ===
print(f"[🧪] Encoded image shape: {bgr_encoded.shape}")
print(f"[🧪] Encoded image dtype: {bgr_encoded.dtype}")
print(f"[🧪] Pixel value range: min={np.min(bgr_encoded)}, max={np.max(bgr_encoded)}")

# === Convert to uint8 if dtype is not valid for saving ===
if bgr_encoded.dtype != np.uint8:
    print("[ℹ️] Converting image to uint8 for saving...")
    bgr_encoded = (bgr_encoded * 255.0).clip(0, 255).astype(np.uint8)

# === Save the result ===
success = cv2.imwrite(output_file, bgr_encoded)
if not success:
    raise IOError(f"[✘] Failed to save image to '{output_file}'")

# === Final output ===
print(f"[✔] Watermark embedded using '{algorithm}' | Input: '{input_file}' → Output: '{output_file}'")
