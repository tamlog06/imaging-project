import cv2
import numpy as np


if __name__ == '__main__':
    # For img1
    src = cv2.imread('img/sample.png')
    dst = cv2.imread('img/1.png')
    mask = cv2.imread('img/mask.png')
    center = (4000, 3000)
    output_normal = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    output_mixed = cv2.seamlessClone(src, dst, mask, center, cv2.MIXED_CLONE)

    cv2.imwrite('output/normal-1.png', output_normal)
    cv2.imwrite('img/mixed-1.png', output_mixed)

    # For img2
    dst = cv2.imread('img/2.png')
    resized_src = cv2.resize(src, (140, 140))
    resized_mask = cv2.resize(mask, (140, 140))
    center = (720, 537)
    output_normal = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.NORMAL_CLONE)
    output_mixed = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.MIXED_CLONE)

    cv2.imwrite('img/normal-2.png', output_normal)
    cv2.imwrite('img/mixed-2.png', output_mixed)

    # For img3
    dst = cv2.imread('img/3.png')
    resized_src = cv2.resize(src, (870, 870))
    resized_mask = cv2.resize(mask, (870, 870))
    center = (944, 572)
    output_normal = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.NORMAL_CLONE)
    output_mixed = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.MIXED_CLONE)

    cv2.imwrite('img/normal-3.png', output_normal)
    cv2.imwrite('img/mixed-3.png', output_mixed)

    # For img4
    dst = cv2.imread('img/4.png')
    resized_src = cv2.resize(src, (680, 680))
    resized_mask = cv2.resize(mask, (680, 680))
    center = (700, 700)
    output_normal = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.NORMAL_CLONE)
    output_mixed = cv2.seamlessClone(resized_src, dst, resized_mask, center, cv2.MIXED_CLONE)

    cv2.imwrite('img/normal-4.png', output_normal)
    cv2.imwrite('img/mixed-4.png', output_mixed)
