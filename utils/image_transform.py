import numpy as np
import cv2

def RGB2YCBRC(img_list):
    out_list = []
    for img in img_list:
        img = img*255.
        img = np.clip(img, 0., 255.)

        im_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        im_ycbcr = np.array(im_ycbcr).astype(np.float32)#  * 255.
        im_y = im_ycbcr[:,:,0]

        out_list.append(im_y)
    out_list = np.array(out_list)
    return out_list