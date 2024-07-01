import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("Notch Filter/SEM.jpg", 0)
    new_img = notch(img)
    fig = plt.figure(figsize = (8, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap = 'gray')
    ax1.title.set_text("Old Image")
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(new_img, cmap = 'gray')
    ax2.title.set_text("New Image")
    ax2.axis('off')
    plt.show()

def notch(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    row, col = img.shape
    nrow, ncol = int(row/2), int(col/2)
    filter = np.ones((row, col, 2), np.uint8)
    filter[nrow, ncol] = 0
    new_shift = dft_shift * filter
    ishift = np.fft.ifftshift(new_shift)
    new_img = cv2.idft(ishift)
    new_img = cv2.magnitude(new_img[:, :, 0], new_img[:, :, 1])
    return new_img

if __name__ == '__main__':
    main()




