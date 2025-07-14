import os
import cv2
import numpy as np
import pywt

# === SETTINGS ===
input_file = 'flow.png'
watermark_text = 'TenCharacs'  # Exactly 10 characters = 80 bits
output_dir = 'D:/WatermarkTests_DWTDCT_QIM'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'flow_qim_dwt_dct_wm.png')  # JPEG-safe output

# === UTILS ===
def text_to_bits(text, max_chars=10):
    text = text.ljust(max_chars, '\x00')[:max_chars]
    return ''.join(format(ord(c), '08b') for c in text)

# === ENCODER ===
def embed_watermark_qim(img, watermark_text, max_chars=10, Q=10, redundancy=5):
    bits = text_to_bits(watermark_text, max_chars)
    b, g, r = cv2.split(img)
    b = np.float32(b)

    # DWT on Blue
    coeffs = pywt.dwt2(b, 'haar')
    LL, (LH, HL, HH) = coeffs
    dct_LL = cv2.dct(LL)
    flat = dct_LL.flatten()

    print(f"[ℹ️] Embedding {len(bits)} bits with redundancy={redundancy}, Q={Q}")
    idx = 0
    for bit in bits:
        for _ in range(redundancy):
            if idx >= len(flat):
                print("[⚠️] Ran out of DCT coefficients!")
                break
            val = flat[idx]
            mod = int(val) % Q
            delta = (1 - mod) if bit == '1' else (-mod)
            flat[idx] += delta
            idx += 1

    dct_LL_mod = flat.reshape(dct_LL.shape)
    LL_mod = cv2.idct(dct_LL_mod)
    b_mod = pywt.idwt2((LL_mod, (LH, HL, HH)), 'haar')
    b_mod = np.clip(b_mod, 0, 255)
    b_mod = cv2.resize(b_mod, (g.shape[1], g.shape[0]))

    final = cv2.merge([b_mod.astype(np.uint8), g, r])
    return final

# === EXECUTE ===
img = cv2.imread(input_file)
if img is None:
    raise FileNotFoundError("Input image not found.")

wm_img = embed_watermark_qim(img, watermark_text)

cv2.imwrite(output_file, wm_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
print(f"[✔] Watermarked image saved to: {output_file}")
