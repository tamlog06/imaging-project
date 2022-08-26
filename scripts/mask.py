import cv2
import numpy as np
import time

drawing = False
ix, iy = -1, -1

def onMouse(event,x,y,flags,param):
    global ix,iy,drawing

    print(event)
    print(drawing)


    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            print('drawing')

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)


if __name__ == '__main__':
    # background = cv2.imread('../image/1.png')
    foreground = cv2.imread('../image/sample.png')

    print(foreground.shape)
    print(foreground)

    mask = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite('../image/mask.png', mask)

    # cv2.imwrite('../image/mask.png', mask)
    # foreground = cv2.resize(foreground, (100, 100))
    # mixed = background
    # mixed[490:490+foreground.shape[0], 660:660+foreground.shape[1]] = foreground

    # cv2.imwrite('../image/source2.png', mixed)
    # foreground = mixed[490:490+foreground.shape[0], 660:660+foreground.shape[1]]
    # grey = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    # ret, binary = cv2.threshold(foreground, 1, 255, cv2.THRESH_BINARY)


    # mask = np.zeros(background.shape, dtype=np.uint8)
    # mask[490:490+foreground.shape[0], 660:660+foreground.shape[1], :] = binary
    # cv2.imwrite('../image/mask2.png', mask)


       # print(cv2.EVENT_LBUTTONDOWN)
    # print(cv2.EVENT_LBUTTONUP)
    # print(cv2.EVENT_MOUSEMOVE)
    # img = cv2.imread('../image/2.png')

    # cv2.namedWindow(winname='image')
    # cv2.setMouseCallback('image', onMouse)

    # while True:
        # cv2.imshow('image', img)
        # if cv2.waitKey(10) == 27:
            # break
    # cv2.destroyAllWindows()



