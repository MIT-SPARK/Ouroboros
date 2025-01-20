import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cv2

cv2.startWindowThread()


def plt_fast_pause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)


def create_image_pair(left, right):
    r = left.shape[0]
    c = left.shape[1] * 2 + 10
    img_out = np.zeros((r, c, 3))
    img_out[: left.shape[0], : left.shape[1], :] = left
    if right is not None:
        img_out[:, left.shape[1] + 10 :, :] = right

    return img_out


def display_image_pair(left, right, window="matches"):
    img = create_image_pair(left, right)
    cv2.imshow(window, img.astype(np.uint8))
    print("going to show")
    cv2.waitKey(10)
    print("showed")
