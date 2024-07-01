import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    old_img = cv2.imread("Low Pass Filter/img.jpg", 0)
    new_img1 = ILPF(old_img, 5)
    new_img2 = ILPF(old_img, 15)
    new_img3 = ILPF(old_img, 30)
    new_img4 = ILPF(old_img, 80)
    new_img5 = ILPF(old_img, 230)
    fig = plt.figure(figsize = (12, 12))
    ax1 = fig.add_subplot(3, 2, 1); ax1.imshow(old_img, cmap = "gray"); ax1.title.set_text("Image"); ax1.axis('off')
    ax2 = fig.add_subplot(3, 2, 2); ax2.imshow(new_img1, cmap = "gray"); ax2.title.set_text("5 radii"); ax2.axis('off')
    ax3 = fig.add_subplot(3, 2, 3); ax3.imshow(new_img2, cmap = "gray"); ax3.title.set_text("15 radii"); ax3.axis('off')
    ax4 = fig.add_subplot(3, 2, 4); ax4.imshow(new_img3, cmap = "gray"); ax4.title.set_text("30 radii"); ax4.axis('off')
    ax5 = fig.add_subplot(3, 2, 5); ax5.imshow(new_img4, cmap = "gray"); ax5.title.set_text("80 radii"); ax5.axis('off')
    ax6 = fig.add_subplot(3, 2, 6); ax6.imshow(new_img5, cmap = "gray"); ax6.title.set_text("230 radii"); ax6.axis('off')
    plt.show()

def ILPF(img, r):  # Ideal Low Pass Filter

    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    nrow, ncol = int(row/2), int(col/2)
    center = [nrow, ncol]
    f = np.zeros((row, col, 2), np.uint8)
    x, y = np.ogrid[:row, :col]
    f_area = np.power(x - center[0], 2) + np.power(y - center[1], 2) <= np.power(r, 2)
    f[f_area] = 1
    new_shift = dft_shift * f
    ishift = np.fft.ifftshift(new_shift)
    new_img = cv2.idft(ishift)
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
    return new_img

if __name__ == "__main__":
    main()

