import cv2
import numpy as np

if __name__ == '__main__':
    sample = cv2.imread('../image/sample.png')
    print(sample.shape)
    back = cv2.imread('../image/1.jpg')
    print(back.shape)

    cropped = back[2200:2200+sample.shape[0]*2, 3000:3000+sample.shape[1]*2, :]

    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)
    print(cropped.shape)
    cv2.imwrite('../image/cropped.png', cropped)
