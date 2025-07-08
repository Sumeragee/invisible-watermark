import os
import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from imwatermark.rivaGan import RivaWatermark

input_file = 'kafka run back.jpg'
output_file = 'watermarked_image.png'
watermark_text = 'SH13'  # Must be exactly 4 chars for rivaGan
algorithm = 'rivaGan'

if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Input image not found: {input_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

print("[ℹ️] Initializing encoder...")
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', watermark_text.encode('utf-8'))

if algorithm == 'rivaGan':
    print("[ℹ️] Loading rivaGan model...")
    encoder._wm_encoder = RivaWatermark(encoder._watermarks, encoder._wmLen)
    encoder._wm_encoder.loadModel()

print("[ℹ️] Encoding in progress...")
bgr_encoded = encoder.encode(bgr, algorithm)

if bgr_encoded is None:
    raise RuntimeError(f"[✘] Failed to encode watermark using '{algorithm}'")

print(f"[🧪] Encoded shape: {bgr_encoded.shape}")
print(f"[🧪] Dtype: {bgr_encoded.dtype}")
print(f"[🧪] Pixel range: {np.min(bgr_encoded)} to {np.max(bgr_encoded)}")

# Fix float output
if bgr_encoded.dtype != np.uint8:
    print("[ℹ️] Cleaning and converting image to uint8...")
    bgr_encoded = np.nan_to_num(bgr_encoded, nan=0.0, posinf=1.0, neginf=0.0)
    bgr_encoded = (bgr_encoded * 255.0).clip(0, 255).astype(np.uint8)

print("[ℹ️] Saving output image...")
success = cv2.imwrite(output_file, bgr_encoded)
if not success:
    raise IOError(f"[✘] Failed to save image to '{output_file}'")

print(f"[✔] Watermark embedded using '{algorithm}' | Input: '{input_file}' → Output: '{output_file}'")
