''' ”Cosmetic” processing is another use of lowpass filtering. 
Following is an application of lowpass filtering to produce a smoother, softer-looking result from a sharp original. 
For human faces, the typical objective is to reduce the sharpness of fine skin lines and small blemishes.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from Gaussian import GLPF


def main():
    img = cv2.imread("Low Pass Filter/Example2_img.jpg", 0)
    new_img1 = GLPF(img, 100)
    new_img2 = GLPF(img, 50)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(img, cmap="gray"); ax1.title.set_text("Image"); ax1.axis('off')
    ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(new_img1, cmap="gray"); ax2.title.set_text("100 radii"); ax2.axis('off')
    ax3 = fig.add_subplot(1, 3, 3); ax3.imshow(new_img2, cmap="gray"); ax3.title.set_text("50 radii"); ax3.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
