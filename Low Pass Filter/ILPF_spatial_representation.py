import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("Low Pass Filter/img2.jpg", 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
mag = 0.5 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) + 1)

 # Ideal Filter
row, col = img.shape
nrow, ncol = int(row / 2), int(col / 2)
center = [nrow, ncol]
r = 5
f = np.zeros((row, col, 2), np.uint8)
x, y = np.ogrid[:row, :col]
f_area = np.power(x - center[0], 2) + np.power(y - center[1], 2) <= np.power(r, 2)
f[f_area] = 1

dft1 = cv2.dft(np.float32(f), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift1 = np.fft.fftshift(dft1)
mag2 = cv2.magnitude(dft_shift1[:, :, 0], dft_shift1[:, :, 1])

new_shift = dft_shift * f
new_shiftmag = 0.5 * np.log((cv2.magnitude(new_shift[:, :, 0], new_shift[:, :, 1])) + 1)
ishift = np.fft.ifftshift(new_shift)
new_img = cv2.idft(ishift)
new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])

fig = plt.figure(figsize=(10, 12))
ax1 = fig.add_subplot(3, 1, 1); ax1.imshow(img, cmap="gray"); ax1.title.set_text("Image"); ax1.axis('off')
ax2 = fig.add_subplot(3, 1, 2); ax2.imshow(mag2, cmap="gray"); ax2.title.set_text("Spatial filter"); ax2.axis('off')
ax3 = fig.add_subplot(3, 1, 3); ax3.imshow(new_img, cmap="gray"); ax3.title.set_text("Convolution of image and filter in the spatial domain."); ax3.axis('off')
plt.show()
