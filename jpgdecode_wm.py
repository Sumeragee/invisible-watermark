# === Decoder with Specific File Name Input and Optional JPEG Handling ===
import os
import cv2
from imwatermark import WatermarkDecoder
from imwatermark.rivaGan import RivaWatermark  # Needed to load model manually

# === Settings ===
input_dir = 'D:/WatermarkTests'
input_file = os.path.join(input_dir, 'flow_wm.jpg')  # <-- Replace as needed
algorithm = 'rivaGan'
max_wm_length = 32  # For RivaGAN: 4 characters → 32 bits

# === Load watermarked image ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Watermarked image not found: {input_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

print(f"[ℹ️] Loaded image shape: {bgr.shape}, dtype: {bgr.dtype}")

# === Optional: Simulate JPEG recompression before decoding ===
simulate_jpeg_read = True  # Toggle this to True/False as needed
jpeg_quality = 75          # Match your encoder if testing robustness

if simulate_jpeg_read:
    print(f"[ℹ️] Simulating JPEG compression before decoding (quality={jpeg_quality})...")
    temp_jpeg_path = os.path.join(input_dir, 'temp_decode_compress.jpg')
    cv2.imwrite(temp_jpeg_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    bgr = cv2.imread(temp_jpeg_path)
    if bgr is None:
        raise IOError(f"[✘] Failed to reload JPEG-compressed image for decoding.")
    else:
        print("[✔] JPEG simulation complete. Proceeding with decoding...")

# === Initialize decoder ===
print("[ℹ️] Initializing decoder...")
decoder = WatermarkDecoder('bytes', max_wm_length)

if algorithm == 'rivaGan':
    print("[ℹ️] Loading RivaGAN model...")
    decoder._wm_decoder = RivaWatermark(None, max_wm_length)
    decoder._wm_decoder.loadModel()

# === Decode watermark ===
print("[ℹ️] Decoding watermark...")
try:
    decoded = decoder.decode(bgr, algorithm)
    decoded_text = decoded.decode('utf-8', errors='replace').strip()

    if not decoded_text or all(c == '\x00' for c in decoded_text):
        print(f"[ℹ️] No invisible watermark detected in '{input_file}'.")
    else:
        print(f"[✔] Decoded watermark from '{input_file}': '{decoded_text}'")

except Exception as e:
    print(f"[✘] Failed to decode watermark — possibly no watermark present or data corrupted.\nDetails: {e}")