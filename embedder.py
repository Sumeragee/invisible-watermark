import cv2
import os
from imwatermark import WatermarkEncoder

# === Output folder ===
output_dir = 'D:/WatermarkTests_DWTDCT'
os.makedirs(output_dir, exist_ok=True)

# === Read input image ===
bgr = cv2.imread('tester.webp')
wm = 'ZeNLeSSZ'  # 8 characters = 64 bits

# === Encode watermark ===
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'dwtDct')

# === Save to D: ===
output_path = os.path.join(output_dir, 'tester_wm.webp')
cv2.imwrite(output_path, bgr_encoded)
print(f"[✔] Watermarked image saved to: {output_path}")
