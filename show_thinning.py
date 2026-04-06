import cv2
import numpy as np

def show_sticks_group(img_otsu, img_thin):
    # Connected components a skeletonon
    num_labels, labels = cv2.connectedComponents(img_thin)
    print("Kontúrok száma:", num_labels)

    component = np.zeros_like(img_thin)
    for label in range(1, num_labels):  # 0 = háttér

        # alap overlay minden iterációban újra
        img_overlay = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2BGR)

        # vastag vonal 5x5
        kernel = np.ones((5, 5), np.uint8)

        # piros színezés
        # component = np.zeros_like(img_thin)
        component[labels == label] = 255

        # vastag vonal kirajzolása
        component_thick = cv2.dilate(component, kernel, iterations=1)
        img_overlay[component_thick == 255] = [0, 0, 255]

        # megjelenítés
        cv2.imshow("Step by step skeleton", img_overlay)
        cv2.waitKey(0)

    return

def show_sticks_all_groups(img_otsu, img_thin):

    img_overlay = cv2.cvtColor(img_otsu, cv2.COLOR_GRAY2BGR)

    # vastag vonal 5x5
    kernel = np.ones((5, 5), np.uint8)
    thick = cv2.dilate(img_thin, kernel, iterations=1)

    rows, cols = np.where(thick == 255)
    img_overlay[rows, cols] = [0, 0, 255]

    return img_overlay

