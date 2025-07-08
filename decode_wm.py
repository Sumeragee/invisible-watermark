# === Decoder That Decodes Most Recent In Directory ===

import os
import cv2
from imwatermark import WatermarkDecoder
from imwatermark.rivaGan import RivaWatermark

input_dir = 'D:/WatermarkTests'
algorithm = 'rivaGan'
max_wm_length = 32

# Get all _wm.png files and find the newest
wm_files = [f for f in os.listdir(input_dir) if f.endswith('_wm.png')]
if not wm_files:
    raise FileNotFoundError("[✘] No watermarked images found.")

latest_file = max(wm_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
input_file = os.path.join(input_dir, latest_file)

print(f"[ℹ️] Decoding latest watermarked image: {latest_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

decoder = WatermarkDecoder('bytes', max_wm_length)

if algorithm == 'rivaGan':
    decoder._wm_decoder = RivaWatermark(None, max_wm_length)
    decoder._wm_decoder.loadModel()

decoded = decoder.decode(bgr, algorithm)
decoded_text = decoded.decode('utf-8') if decoded else "[✘] No watermark found"
print(f"[✔] Decoded watermark: {decoded_text}")
