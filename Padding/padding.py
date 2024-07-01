import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("/Users/ananyasharma/Desktop/VS Code/fourier analysis/IPLF1.jpg", 0)
A, B = img.shape
C, D = 500, 500
nrow, ncol = int(C / 2), int(D / 2)
center = [nrow, ncol]
r = 355
f = np.zeros((C, D), np.uint8)
x, y = np.ogrid[:C, :D]
f_area = np.power(x - center[0], 2) + np.power(y - center[1], 2) <= np.power(r, 2)
f[f_area] = 1
padding_size1 = int((A + C - 1) / 2)
padding_size2 = int((B + D - 1) / 2)

# padding image
img = np.pad(
    img, ((0, padding_size1), (0, padding_size1)), mode="constant", constant_values=0
)

# padding filter
f = np.pad(
    f, ((0, padding_size2), (0, padding_size2)), mode="constant", constant_values=0
)

# Applying filtering with padding
img = np.fft.fftshift(np.fft.fft2(img))
dft = img * f
ishift = np.fft.ifftshift(dft)
new_img = np.fft.ifft2(ishift)
new_img = np.real(new_img)
img1 = cv2.normalize(new_img, None, -1, 1, norm_type=cv2.NORM_MINMAX)
cv2.imshow("window1", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
