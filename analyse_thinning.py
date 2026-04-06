import cv2
import numpy as np

def get_end_point_mask(img_thin):

    # minden nullára állítunk
    endpoints_mask = np.zeros_like(img_thin, dtype=np.uint8)

    # Endpoint detektálás 8-neighborhood alapján
    rows, cols = np.where(img_thin == 255)  # csak fehér pixelek

    for r, c in zip(rows, cols):
        # 8 szomszéd koordinátái, 3x3-as ablak a pixel körül, tartalmazza a középpontot is
        neighbors = img_thin[r - 1:r + 2, c - 1:c + 2]

        # összes fehér szomszéd (kivéve a középpont)
        count = np.sum(neighbors == 255) - 1

        if count == 1:
            endpoints_mask[r, c] = 255  # ez egy végpont

    return endpoints_mask

def get_overlay_with_end_point_mask(img_otsu, img_thin):

    # végpontok keresése
    endpoints_mask = get_end_point_mask(img_thin)

    # Overlay készítése a vizualizációhoz
    img_overlay = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2BGR)

    # vastag pont rajzolása
    kernel = np.ones((7, 7), np.uint8)
    endpoints_thick = cv2.dilate(endpoints_mask, kernel, iterations=1)

    # Endpointokat kék pontokkal rajzoljuk
    img_overlay[endpoints_thick == 255] = [255, 0, 0]  # kék

    return img_overlay

def get_number_of_sticks(img_thin):
    # végpontok keresése
    endpoints_mask = get_end_point_mask(img_thin)

    # végpontok keresése
    num_endpoints = np.sum(endpoints_mask == 255)

    return num_endpoints // 2