import cv2
import numpy as np


def embossing(img):
    femboss = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray16 = np.int16(gray)
    embossing = np.uint8(np.clip(cv2.filter2D(gray16, -1, femboss) + 128, 0, 255))

    return embossing