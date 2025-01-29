import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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


def display_image_pair(left, right, window="place_matches", show=True):
    img = create_image_pair(left, right)
    cv2.imshow(window, img.astype(np.uint8))
    if show:
        cv2.waitKey(10)


def add_kp_matches(img, kp_left, kp_right, color=(0, 255, 0)):
    offset = img.shape[1] // 2
    for (l_x, l_y), (r_x, r_y) in zip(kp_left, kp_right):
        cv2.line(img, (l_x, l_y), (r_x + offset, r_y), color, 1)


def create_kp_match_pair(vlc_left, vlc_right, kp_left, kp_right):
    img = vlc_left.image.rgb
    if img.ndim == 2 or img.shape[2] == 1:
        left_color = cv2.cvtColor(
            vlc_left.image.rgb.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
        right_color = cv2.cvtColor(
            vlc_right.image.rgb.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
    else:
        left_color = vlc_left.image.rgb.astype(np.uint8)
        right_color = vlc_right.image.rgb.astype(np.uint8)
    img = create_image_pair(left_color, right_color)
    add_kp_matches(img, kp_left.astype(int), kp_right.astype(int))
    return img


def display_kp_match_pair(
    vlc_left, vlc_right, kp_left, kp_right, window="kp_matches", show=True
):
    img = create_kp_match_pair(vlc_left, vlc_right, kp_left, kp_right)
    cv2.imshow(window, img / 255)
    if show:
        cv2.waitKey(10)
