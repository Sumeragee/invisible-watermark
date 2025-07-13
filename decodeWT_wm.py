# === DWT-DCT Decoder for up to 10-character invisible watermark ===
import os
import cv2
import numpy as np
import pywt

# === Settings ===
input_dir = 'D:/WatermarkTests_DWTDCT'
input_file = os.path.join(input_dir, 'agnes_dwt_dct_wm.jpg')
max_watermark_length = 10  # 10 characters = 80 bits

# === Binary to text conversion ===
def bits_to_text(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

# === Extract watermark from image ===
def extract_watermark_dwt_dct(img, max_chars=10):
    bit_len = max_chars * 8  # e.g. 10 chars = 80 bits

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Apply 2D DWT
    coeffs = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Apply DCT to LL
    dct_LL = cv2.dct(LL)

    # Extract LSBs from DCT coefficients
    flat = dct_LL.flatten()
    bits = ''
    for i in range(bit_len):
        if i < len(flat):
            bits += str(int(flat[i]) % 2)
        else:
            break

    return bits_to_text(bits)

# === Load image ===
if not os.path.exists(input_file):
    raise FileNotFoundError(f"[✘] Watermarked image not found: {input_file}")

bgr = cv2.imread(input_file)
if bgr is None:
    raise ValueError(f"[✘] Failed to load image: {input_file}")

# === Decode ===
print("[ℹ️] Decoding watermark from DWT-DCT image...")
try:
    decoded_text = extract_watermark_dwt_dct(bgr, max_chars=max_watermark_length)

    if not decoded_text.strip() or all(c == '\x00' for c in decoded_text):
        print(f"[ℹ️] No watermark detected in '{input_file}'.")
    else:
        print(f"[✔] Decoded watermark (max 10 chars): '{decoded_text.strip()}'")

except Exception as e:
    print(f"[✘] Failed to decode watermark — possibly no watermark present or format is wrong.\nDetails: {e}")
