import cv2
import numpy as np

def add_point_noise(img_in, percentage, value):
    noise_res = np.copy(img_in)
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)

    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])

        if img_in.ndim == 2:
            noise_res[j, i] = value

        if img_in.ndim == 3:
            noise_res[j, i] = [value, value, value]

    return noise_res

def add_salt_and_pepper_noise(img_in, percentage1, percentage2):
    n = add_point_noise(img_in, percentage1, 255)   # Só
    n2 = add_point_noise(n, percentage2, 0)         # Bors

    return n2

def create_img_with_noise(im_src):
    # Konvertálás HSV színtérbe
    img_hsv = cv2.cvtColor(im_src, cv2.COLOR_BGR2HSV)

    # Csatornák szétválasztása
    H, S, V = cv2.split(img_hsv)

    # Só-bors zaj hozzáadása csatornánként
    H_noisy = add_salt_and_pepper_noise(H, percentage1=0.01, percentage2=0.01)
    S_noisy = add_salt_and_pepper_noise(S, percentage1=0.01, percentage2=0.01)
    V_noisy = add_salt_and_pepper_noise(V, percentage1=0.02, percentage2=0.02)

    # Csatornák egyesítése
    img_hsv_noisy = cv2.merge([H_noisy, S_noisy, V_noisy])

    # Vissza BGR formátumba
    img_noisy = cv2.cvtColor(img_hsv_noisy, cv2.COLOR_HSV2BGR)

    return img_noisy
