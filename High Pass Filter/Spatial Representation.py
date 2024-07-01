import cv2
import numpy as np
from matplotlib import pyplot as plt

row, col = 500, 500
nrow, ncol = int(row / 2), int(col / 2)
center = [nrow, ncol]
r = 5

# Ideal high pass filter
H1 = np.ones((row, col, 2), np.uint8)
x, y = np.ogrid[:row, :col]
H1_area = np.power(x - center[0], 2) + np.power(y - center[1], 2) <= np.power(r, 2)
H1[H1_area] = 0
dft1 = cv2.dft(np.float32(H1), flags=cv2.DFT_COMPLEX_OUTPUT)
shift1 = np.fft.fftshift(dft1)
mag1 = 0.5 * np.log((cv2.magnitude(shift1[:, :, 0], shift1[:, :, 1])) + 1)

# Butterworth high pass filter
H2 = np.zeros((row, col, 2), np.float32)
n = 2
for u in range(row):
    for v in range(col):
        D = np.sqrt(np.power(u - center[0], 2) + np.power(v - center[1], 2))
        H2[u, v] = 1 / (1 + (r / D) ** (2 * n))
dft2 = cv2.dft(np.float32(H2), flags=cv2.DFT_COMPLEX_OUTPUT)
shift2 = np.fft.fftshift(dft2)
mag2 = 0.5 * np.log((cv2.magnitude(shift2[:, :, 0], shift2[:, :, 1])) + 1)

# Gaussian high pass filter
H3 = np.zeros((row, col, 2), np.float32)
for u in range(row):
    for v in range(col):
        D = np.sqrt(np.power(u - center[0], 2) + np.power(v - center[1], 2))
        H3[u, v] = 1 - np.exp(-np.power(D, 2) / (2 * np.power(r, 2)))
dft3 = cv2.dft(np.float32(H3), flags=cv2.DFT_COMPLEX_OUTPUT)
shift3 = np.fft.fftshift(dft3)
mag3 = 0.5 * np.log((cv2.magnitude(shift3[:, :, 0], shift3[:, :, 1])) + 1)

# Plotting spatial domain
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(mag1, cmap="gray")
ax1.title.set_text("Ideal high pass filter")
ax1.axis("off")
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(mag2, cmap="gray")
ax2.title.set_text("Butterworth high pass filter")
ax2.axis("off")
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(mag3, cmap="gray")
ax3.title.set_text("Gaussian high pass filter")
ax3.axis("off")
plt.show()
