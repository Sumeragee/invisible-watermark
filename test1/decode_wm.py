import cv2
from imwatermark import WatermarkDecoder

# === Settings ===
input_file = 'watermarked_image.png'
algorithm = 'dwtDct'
max_wm_length = 32  # Adjust based on how long your watermark is

# === Load image ===
bgr = cv2.imread(input_file)

# === Decode watermark ===
decoder = WatermarkDecoder('bytes', max_wm_length)
decoded = decoder.decode(bgr, algorithm)

# === Log ===
print(f"[✔] Decoded watermark from '{input_file}' using '{algorithm}': {decoded.decode('utf-8')}")

