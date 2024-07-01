import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread("UTK.jpg", 0)
img2 = cv2.imread("T.jpg", 0)

# padding image 1
img1 = np.pad(img1, ((0, 42), (0, 42)), mode="constant", constant_values=0)

# padding image 2
img2 = np.pad(img2, ((0, 256), (0, 260)), mode="constant", constant_values=0)

# Finding complex conjugate of Fourier Transform of image 1
complex_img1 = np.conj(np.fft.fftshift(np.fft.fft2(img1)))

# Finding Fourier Transform of image 2
dft_img2 = np.fft.fftshift(np.fft.fft2(img2))

# Correlating image 1 amd image 2
img = complex_img1 * dft_img2
img = np.real(np.fft.ifft2(np.fft.ifftshift(img)))
img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("window1", img1)
cv2.imshow("window2", img2)
cv2.imshow("window3", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
