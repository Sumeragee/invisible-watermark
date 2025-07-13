# === Encoder/Embedder of INV WM to IMG with JPEG Compression Simulation ===
# === RivaGAN only supports 32-bit payloads (4 characters) ===

import os
import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from imwatermark.rivaGan import RivaWatermark

# === Input image and watermark ===
input_file = 'agnes.jpg'        # <-- Input file
watermark_text = 'EDRS'        # Must be exactly 4 characters
algorithm = 'rivaGan'

# === Output directory ===
output_dir = 'D:/WatermarkTests'
os.makedirs(output_dir, exist_ok=True)

# === File name preparation ===
base_name = os.path.basename(input_file)
name_no_ext, ext = os.path.splitext(base_name)
output_file = os.path.join(output_dir, f"{name_no_ext}_wm{ext}")
dummy_test_path = os.path.join(output_dir, 'test_save_dummy.png')

# === Load input image ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Input image not found: {input_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

# === Initialize encoder ===
print("[ℹ️] Initializing encoder...")
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', watermark_text.encode('utf-8'))

if algorithm == 'rivaGan':
    print("[ℹ️] Loading RivaGAN model...")
    encoder._wm_encoder = RivaWatermark(encoder._watermarks, encoder._wmLen)
    encoder._wm_encoder.loadModel()

# === Encode watermark ===
print("[ℹ️] Encoding in progress...")
bgr_encoded = encoder.encode(bgr, algorithm)

if bgr_encoded is None:
    raise RuntimeError(f"[✘] Failed to encode watermark using '{algorithm}'")

print(f"[🧪] Encoded shape: {bgr_encoded.shape}")
print(f"[🧪] Dtype: {bgr_encoded.dtype}")
print(f"[🧪] Pixel range: {np.min(bgr_encoded)} to {np.max(bgr_encoded)}")

# === Fix float output if needed ===
if bgr_encoded.dtype != np.uint8:
    print("[ℹ️] Converting image to uint8...")
    bgr_encoded = np.nan_to_num(bgr_encoded, nan=0.0, posinf=1.0, neginf=0.0)
    bgr_encoded = (bgr_encoded * 255.0).clip(0, 255).astype(np.uint8)

# === Preview image ===
print("[ℹ️] Previewing image (press any key to continue)...")
cv2.imshow("Preview: Watermarked Image", bgr_encoded)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Dummy test image save ===
print("[ℹ️] Testing if OpenCV can save at all...")
dummy_image = np.full_like(bgr_encoded, 127)
dummy_saved = cv2.imwrite(dummy_test_path, dummy_image)
if dummy_saved:
    print(f"[✔] Dummy image saved successfully to {dummy_test_path}")
else:
    raise IOError("[✘] OpenCV failed to save a basic dummy image. Check path or permissions.")

# === Optional: Simulate JPEG compression before saving final output ===
simulate_jpeg = True
jpeg_quality = 75

if simulate_jpeg:
    print(f"[ℹ️] Simulating JPEG compression at quality={jpeg_quality}...")
    jpeg_temp_path = os.path.join(output_dir, f"{name_no_ext}_temp_compress.jpg")
    cv2.imwrite(jpeg_temp_path, bgr_encoded, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    bgr_encoded = cv2.imread(jpeg_temp_path)
    if bgr_encoded is None:
        raise IOError(f"[✘] Failed to reload JPEG-compressed image: {jpeg_temp_path}")
    else:
        print(f"[✔] JPEG compression simulation complete. Reloaded image shape: {bgr_encoded.shape}")

# === Final save of watermarked image ===
print("[ℹ️] Saving final watermarked image...")
success = cv2.imwrite(output_file, bgr_encoded)
if not success:
    raise IOError(f"[✘] Failed to save image to '{output_file}'")

print(f"[✔] Watermark embedded using '{algorithm}' | Input: '{input_file}' → Output: '{output_file}'")
