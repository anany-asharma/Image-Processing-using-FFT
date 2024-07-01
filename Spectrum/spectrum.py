import cv2 
import numpy as np
from matplotlib import pyplot as plt

def main():
    img = cv2.imread("Spectrum/Rectangle.jpg", 0)
    # Following code is used to create a white rectangle(20 X 40) superimposed on a black rectangle(512 X 512)
    '''img=np.zeros((512,512,3),np.uint8)*255
    img=cv2.rectangle(img=new_img,pt1=(236,246),pt2=(276,266), color=(255,255,255),thickness=-1)
    cv2.imwrite("Rectangle.jpg",img)'''


    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # cv2.dft() function computes the discrete FT of the image. 
    # This function takes in the image as an argument and returns the FT as a NumPy array.

    dft_shift = np.fft.fftshift(dft)
    # np.fft.fftshift() function shifts the zero-frequency component of the FT to the center of the array.

    spectrum = 0.5 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) + 1)

    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, cmap = 'gray')
    ax1.title.set_text("Rectangle")
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(spectrum, cmap = 'gray')
    ax2.title.set_text("FFT of rectangle")
    ax2.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

