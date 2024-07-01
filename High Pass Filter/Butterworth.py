import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    old_img = cv2.imread(
        "Low Pass Filter/img.jpg", 0
    )
    new_img1 = BHPF(old_img, 15)
    new_img2 = BHPF(old_img, 30)
    new_img3 = BHPF(old_img, 80)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1); ax1.imshow(old_img, cmap="gray"); ax1.title.set_text("Image"); ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2); ax2.imshow(new_img1, cmap="gray"); ax2.title.set_text("15 radii"); ax2.axis("off")
    ax3 = fig.add_subplot(2, 2, 3); ax3.imshow(new_img2, cmap="gray"); ax3.title.set_text("30 radii"); ax3.axis("off")
    ax4 = fig.add_subplot(2, 2, 4); ax4.imshow(new_img3, cmap="gray"); ax4.title.set_text("80 radii"); ax4.axis("off")
    plt.show()


def BHPF(img, r):

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    nrow, ncol = int(row / 2), int(col / 2)
    center = [nrow, ncol]
    n = 2
    f = np.zeros((row, col, 2), np.float32)
    for x in range(row):
        for y in range(col):
            D = np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))
            f[x, y] = 1 / (1 + (r / D) ** (2 * n))
    new_shift = dft_shift * f
    ishift = np.fft.ifftshift(new_shift)
    new_img = cv2.idft(ishift)
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
    return new_img


if __name__ == '__main__':
    main()