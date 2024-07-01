'''Let there be a sample of text of poor resolution in which many of the characters are broken. 
Humans can fill these gaps visually without difficult but a machine recognition system has difficulties reading broken characters. 
The approach used most often to handle this problem is to bridge small gaps in the input image by blurring it.'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from Gaussian import GLPF

def main():
    img = cv2.imread("Low Pass Filter/Example1_img.jpg", 0)
    new_img = GLPF(img, 80)
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(img, cmap="gray"); ax1.title.set_text("Image"); ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(new_img, cmap="gray"); ax2.title.set_text("New Image"); ax2.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
