import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    old_img = cv2.imread("Homomorphic Filtering/img_homo.jpg", 0)
    new_img1 = homo(old_img, 230)
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(old_img, cmap="gray")
    ax1.title.set_text("Image")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(new_img1, cmap="gray")
    ax2.title.set_text("Image after filter")
    ax2.axis("off")
    plt.show()


def homo(img, r):
    img = np.log1p(np.float64(img), dtype=np.float64)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    nrow, ncol = int(row / 2), int(col / 2)
    center = [nrow, ncol]
    gh = 2.0
    gl = 0.5
    h = np.zeros((row, col), np.float32)
    for x in range(row):
        for y in range(col):
            D = np.sqrt(np.power(x - center[0], 2) + np.power(y - center[1], 2))
            h[x, y] = 1 - np.exp(-np.power(D, 2) / (2 * np.power(r, 2)))
    h = (gh - gl) * h + gl
    new_shift = dft_shift * h
    ishift = np.fft.ifftshift(new_shift)
    new_img = np.fft.ifft2(ishift)
    new_img = np.real(new_img)
    new_img = np.expm1(new_img, dtype=np.float64)
    new_img = cv2.normalize(new_img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    return new_img


if __name__ == '__main__':
    main()
