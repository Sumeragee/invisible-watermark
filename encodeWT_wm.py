# === DWT-DCT Watermark Encoder (color-safe, max 10 characters) ===
import os
import cv2
import numpy as np
import pywt

# === Input Settings ===
input_file = 'flow.jpg'
watermark_text = 'TenCharacs'  # Must be exactly 10 characters
output_dir = 'D:/WatermarkTests_DWTDCT'
os.makedirs(output_dir, exist_ok=True)

base_name = os.path.basename(input_file)
name_no_ext, ext = os.path.splitext(base_name)
output_file = os.path.join(output_dir, f"{name_no_ext}_dwt_dct_wm{ext}")

# === Convert watermark to binary ===
def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

# === Embed into Blue channel only ===
def embed_watermark_dwt_dct_color(img, watermark_text, max_chars=10):
    # Enforce exactly max_chars
    if len(watermark_text) < max_chars:
        watermark_text = watermark_text.ljust(max_chars, '\x00')
    elif len(watermark_text) > max_chars:
        watermark_text = watermark_text[:max_chars]

    watermark_bits = text_to_bits(watermark_text)

    # Split color channels
    b, g, r = cv2.split(img)
    b = np.float32(b)

    # DWT → DCT on Blue channel
    coeffs = pywt.dwt2(b, 'haar')
    LL, (LH, HL, HH) = coeffs
    dct_LL = cv2.dct(LL)

    # Embed bits using LSB substitution
    flat = dct_LL.flatten()
    for i in range(min(len(watermark_bits), len(flat))):
        flat[i] = flat[i] - (flat[i] % 2) + int(watermark_bits[i])
    dct_LL_mod = flat.reshape(dct_LL.shape)

    # Rebuild Blue channel
    LL_mod = cv2.idct(dct_LL_mod)
    watermarked_b = pywt.idwt2((LL_mod, (LH, HL, HH)), 'haar')
    watermarked_b = np.clip(watermarked_b, 0, 255).astype(np.uint8)

    # Merge with untouched G and R channels
    final = cv2.merge([watermarked_b, g, r])
    return final

# === Load and process ===
print("[ℹ️] Loading input image...")
img = cv2.imread(input_file)
if img is None:
    raise FileNotFoundError(f"Image not found: {input_file}")

print("[ℹ️] Embedding watermark...")
watermarked_img = embed_watermark_dwt_dct_color(img, watermark_text)

# === Preview result ===
cv2.imshow("Watermarked Image", watermarked_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Save result ===
if cv2.imwrite(output_file, watermarked_img):
    print(f"[✔] Watermarked image saved to: {output_file}")
else:
    print("[✘] Failed to save output image.")
