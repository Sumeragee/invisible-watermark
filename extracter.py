import cv2
import os
from imwatermark import WatermarkDecoder

# === Path to watermarked image ===
input_path = 'D:/WatermarkTests_DWTDCT/tester_wm.webp'

# === Read image ===
bgr = cv2.imread(input_path)
if bgr is None:
    raise FileNotFoundError(f"[✘] Cannot read image from {input_path}")

# === Decode watermark ===
decoder = WatermarkDecoder('bytes', 64)  # 8 chars = 64 bits

try:
    watermark = decoder.decode(bgr, 'dwtDct')
    decoded = watermark.decode('utf-8')
    print(f"[✔] Decoded watermark: '{decoded}'")
except UnicodeDecodeError:
    print("[⚠️] Failed to decode watermark — invalid UTF-8 sequence.")
    print("[🧪] Raw watermark bytes:", list(watermark))
