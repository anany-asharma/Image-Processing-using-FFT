
import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    row, col = 100, 100
    nrow, ncol = int(row/2), int(col/2)
    center = [nrow, ncol]
    r = 5

    H = np.zeros((row, col, 2), np.float32)
    for u in range(row):
        for v in range(col):
            H[u,v] = (-np.power(u - center[0], 2) + np.power(v - center[1], 2))
    dft = cv2.dft(np.float32(H), flags = cv2.DFT_COMPLEX_OUTPUT)
    shift = np.fft.fftshift(dft)
    mag = 0.5 * np.log((cv2.magnitude(shift[:, :, 0], shift[:, :, 1])) + 1)

    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.imshow(mag, cmap = "gray")
    ax1.title.set_text("Spatial domain")
    ax1.axis("off")
    plt.show()


if __name__ == '__main__':
    main()

