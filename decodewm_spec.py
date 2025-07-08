# === Decoder with Specific File Name Input ===
import os
import cv2
from imwatermark import WatermarkDecoder
from imwatermark.rivaGan import RivaWatermark  # Needed to load model manually

# === Settings ===
input_dir = 'D:/WatermarkTests'
input_file = os.path.join(input_dir, 'fufu_wm.png')
algorithm = 'rivaGan'         # Match your encoding algorithm
max_wm_length = 32            # SH13 = 4 characters

# === Load image ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Watermarked image not found: {input_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

# === Initialize decoder ===
print("[ℹ️] Initializing decoder...")
decoder = WatermarkDecoder('bytes', max_wm_length)

# === Manually load rivaGan model ===
if algorithm == 'rivaGan':
    print("[ℹ️] Loading rivaGan model for decoder...")
    decoder._wm_decoder = RivaWatermark(None, max_wm_length)
    decoder._wm_decoder.loadModel()

# === Decode watermark ===
print("[ℹ️] Decoding watermark...")
try:
    decoded = decoder.decode(bgr, algorithm)
    decoded_text = decoded.decode('utf-8').strip()

    if not decoded_text or all(c == '\x00' for c in decoded_text):
        print(f"[ℹ️] No invisible watermark detected in '{input_file}'.")
    else:
        print(f"[✔] Decoded watermark from '{input_file}' using '{algorithm}': {decoded_text}")

except Exception as e:
    print(f"[✘] Failed to decode watermark — possibly no watermark present. Details: {e}")
