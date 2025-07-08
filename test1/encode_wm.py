import cv2
from imwatermark import WatermarkEncoder

# === Settings ===
input_file = 'kafka run back.jpg'
output_file = 'watermarked_image.jpg'
watermark_text = 'Goldshi'
algorithm = 'dwtDct'

# === Load the image ===
bgr = cv2.imread(input_file)

# === Encode watermark ===
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', watermark_text.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, algorithm)

# === Save result ===
cv2.imwrite(output_file, bgr_encoded)

# === Log ===
print(f"[✔] Watermark embedded using '{algorithm}' | Original: '{input_file}' → Output: '{output_file}'")