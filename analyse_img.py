import numpy as np

def invert_img(img_bin):
    # kép invertálása, ha fehér a háttér
    inverted = False
    if np.sum(img_bin == 255) > np.sum(img_bin == 0):
        img_bin = 255 - img_bin
        inverted = True

    return inverted, img_bin