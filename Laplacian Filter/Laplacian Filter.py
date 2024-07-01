import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Laplacian Filter/moon.jpg', 0)
img = img / 255
F = np.fft.fftshift(np.fft.fft2(img))

# Laplacian Filter
row, col = F.shape
nrow, ncol = int(row/2), int(col/2)
center = [nrow, ncol]
H = np.zeros((row, col), dtype=np.float32)
for u in range(row):
    for v in range(col):
        H[u,v] = -4 * np.pi * np.pi * ((u - center[0]) ** 2 + (v - center[1]) ** 2)
        
# Laplacian image
Lap = H * F
Lap = np.fft.ifftshift(Lap)
Lap = np.real(np.fft.ifft2(Lap))


# convert the Laplacian Image value into range [-1,1]
ScaledImg = cv2.normalize(Lap, None, -1, 1, cv2.NORM_MINMAX)

# image ehancement
g = img - ScaledImg
g = np.clip(g, 0, 1)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1); ax1.axis('off'); ax1.imshow(img, cmap="gray") ; ax1.title.set_text("Image")
ax2 = fig.add_subplot(2, 2, 2); ax2.axis('off'); ax2.imshow(H, cmap="gray"); ax2.title.set_text("After laplacian filter")
ax3 = fig.add_subplot(2, 2, 3); ax3.axis('off'); ax3.imshow(ScaledImg, cmap="gray"); ax3.title.set_text("Laplacian image scaled")
ax4 = fig.add_subplot(2, 2, 4); ax4.axis('off'); ax4.imshow(g, cmap="gray"); ax4.title.set_text("Enhanced image")
plt.show()

