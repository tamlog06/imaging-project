import cv2
import numpy as np

if __name__ == '__main__':
    background = cv2.imread('../image/1.jpg')
    foreground = cv2.imread('../image/sample.png')
    grey = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)

    mask = np.zeros(background.shape[:2], dtype=np.uint8)
    mask[2700:2700+foreground.shape[0], 3500:3500+foreground.shape[1]] = binary
    cv2.imwrite('../image/mask.png', mask)

    source = np.zeros(background.shape, dtype=np.uint8)
    source[2700:2700+foreground.shape[0], 3500:3500+foreground.shape[1], :] = foreground
    cv2.imwrite('../image/source.png', source)

