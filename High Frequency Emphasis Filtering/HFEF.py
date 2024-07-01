import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread("High Frequency Emphasis Filtering/img_HPEF.jpg", 0)
    new_img1 = BHPF(img)
    new_img2 = HFEF(img)
    img_back = cv2.normalize(new_img2, None, 0, 255, cv2.NORM_MINMAX)
    new_img3 = cv2.equalizeHist(img_back.astype("uint8"))
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap="gray")
    ax1.axis("off")
    ax1.title.set_text("Image")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(new_img1, cmap="gray")
    ax2.axis("off")
    ax2.title.set_text("Butterworth highpass filtering")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(new_img2, cmap="gray")
    ax3.axis("off")
    ax3.title.set_text("High-frequency emphasis filtering")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(new_img3, cmap="gray")
    ax4.axis("off")
    ax4.title.set_text("After Histogram Equalization")
    plt.show()


def BHPF(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    n = 2
    r = row * 5 / 100
    f = np.zeros((row, col, 2), np.float32)
    for x in range(row):
        for y in range(col):
            D = np.sqrt(np.power(x - int(row / 2), 2) + np.power(y - int(col / 2), 2))
            f[x, y] = 1 / (1 + (r / D) ** (2 * n))
    new_shift = dft_shift * f
    ishift = np.fft.ifftshift(new_shift)
    new_img = cv2.idft(ishift)
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
    return new_img


def HFEF(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    n = 2
    r = row * 5 / 100
    f = np.zeros((row, col, 2), np.float32)
    h = np.zeros((row, col, 2), np.float32)
    for x in range(row):
        for y in range(col):
            D = np.sqrt(np.power(x - int(row / 2), 2) + np.power(y - int(col / 2), 2))
            f[x, y] = 1 / (1 + (r / D) ** (2 * n))
            h[x, y] = 0.5 + 2.0 * f[x, y]
    new_shift = dft_shift * h
    ishift = np.fft.ifftshift(new_shift)
    new_img = cv2.idft(ishift)
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
    return new_img


if __name__ == '__main__':
    main()
