import os
import cv2
from imwatermark import WatermarkEncoder

input_file = 'kafka run back.jpg'
output_file = 'watermarked_image.jpg'
watermark_text = 'SH13'
algorithm = 'rivaGan'  # or 'dwtDct'

# Check if input exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Input image not found: {input_file}")

# Load image
bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

# Encode
print("[ℹ️] Encoding in progress...")
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', watermark_text.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, algorithm)

if bgr_encoded is None:
    raise RuntimeError(f"[✘] Failed to encode watermark using algorithm '{algorithm}'")

# Save
success = cv2.imwrite(output_file, bgr_encoded)
if not success:
    raise IOError(f"[✘] Failed to save image to '{output_file}'")

print(f"[✔] Watermark embedded using '{algorithm}' | Input: '{input_file}' → Output: '{output_file}'")