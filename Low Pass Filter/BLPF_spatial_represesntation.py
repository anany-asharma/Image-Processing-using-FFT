import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    mag1 = BLPF(1)
    mag2 = BLPF(2)
    mag3 = BLPF(5)
    mag4 = BLPF(20)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(mag1, cmap="gray")
    ax1.title.set_text("Order 1")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(mag2, cmap="gray")
    ax2.title.set_text("Order 2")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(mag3, cmap="gray")
    ax3.title.set_text("Order 5")
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(mag4, cmap="gray")
    ax4.title.set_text("Order 20")
    plt.show()


def BLPF(n):
    row, col = 500, 500
    nrow, ncol = int(row / 2), int(col / 2)
    center = [nrow, ncol]
    r = 5
    H = np.zeros((row, col, 2), np.float32)
    for u in range(row):
        for v in range(col):
            D = np.sqrt(np.power(u - center[0], 2) + np.power(v - center[1], 2))
            H[u, v] = 1 / (1 + (D / r) ** (2 * n))
    dft = cv2.dft(np.float32(H), flags=cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(dft)
    mag = 0.5 * np.log((cv2.magnitude(shift[:, :, 0], shift[:, :, 1])) + 1)
    return mag


if __name__ == '__main__':
    main()
