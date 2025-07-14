import os
import cv2
import numpy as np
import pywt

# === SETTINGS ===
input_file = 'D:/WatermarkTests_DWTDCT_QIM/flow_qim_dwt_dct_wm.png'
max_chars = 10
Q = 10               # Quantization divisor (must match encoder)
redundancy = 5       # Repetition count (must match encoder)

# === UTILS ===
def bits_to_text(bits):
    return ''.join(chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)).rstrip('\x00')

# === DECODER ===
def extract_watermark_qim(img, max_chars=10, Q=10, redundancy=5):
    total_bits = max_chars * 8
    b, g, r = cv2.split(img)
    b = np.float32(b)

    coeffs = pywt.dwt2(b, 'haar')
    LL, (LH, HL, HH) = coeffs
    dct_LL = cv2.dct(LL)
    flat = dct_LL.flatten()

    bits = ''
    idx = 0
    for bit_num in range(total_bits):
        bit_votes = []
        for _ in range(redundancy):
            if idx >= len(flat):
                break
            val = flat[idx]
            mod = int(round(val)) % Q
            vote = 1 if mod == 1 else 0
            bit_votes.append(vote)
            idx += 1

        # Majority voting
        if bit_votes:
            bit = str(int(sum(bit_votes) >= (len(bit_votes) // 2 + 1)))
            bits += bit

    return bits_to_text(bits)

# === EXECUTE ===
img = cv2.imread(input_file)
if img is None:
    raise FileNotFoundError("Watermarked image not found.")

print("[ℹ️] Decoding watermark using QIM with voting...")
wm_text = extract_watermark_qim(img, max_chars=max_chars, Q=Q, redundancy=redundancy)
print(f"[✔] Decoded watermark: '{wm_text}'")
