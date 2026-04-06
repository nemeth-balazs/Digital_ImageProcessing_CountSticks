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

def add_salt_and_pepper_noise_with_noise_level(img_in, percentage1, percentage2):
    n = add_point_noise(img_in, percentage1, 255)   # Só
    n2 = add_point_noise(n, percentage2, 0)         # Bors

    return n2

def add_salt_and_pepper_noise(im_src, noise_level):
    # Konvertálás HSV színtérbe
    img_hsv = cv2.cvtColor(im_src, cv2.COLOR_BGR2HSV)

    # Csatornák szétválasztása
    H, S, V = cv2.split(img_hsv)

    # Só-bors zaj hozzáadása csatornánként
    noise = noise_level * 0.01
    V_noisy = add_salt_and_pepper_noise_with_noise_level(V, percentage1=noise, percentage2=noise)

    # Csatornák egyesítése
    img_hsv_noisy = cv2.merge([H, S, V_noisy])

    # Vissza BGR formátumba
    img_noisy = cv2.cvtColor(img_hsv_noisy, cv2.COLOR_HSV2BGR)

    return img_noisy

def add_gaussian_noise(img, noise_level):
        # RGB → HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

        # csatornák szétválasztása
        h, s, v = cv2.split(img_hsv)

        # Gaussian zaj csak a V-re
        noise = np.random.normal(0, noise_level * 3, v.shape).astype(np.float32)
        v_noisy = v + noise
        v_noisy = np.clip(v_noisy, 0, 255)

        # vissza HSV-be
        img_hsv_noisy = cv2.merge([h, s, v_noisy])

        # HSV → RGB
        img_rgb_noisy = cv2.cvtColor(img_hsv_noisy.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return img_rgb_noisy

def create_img_with_noise(im_src, noise_level):

    img_noisy = add_gaussian_noise(im_src, noise_level)
    img_noisy = add_salt_and_pepper_noise(img_noisy, noise_level)

    return img_noisy

def get_noise_level():
    default_value = 1

    while True:
        try:
            user_input = input(f"Add meg a zaj szintet (1-5) [default: {default_value}]: ")
            if user_input.strip() == "":
                return default_value

            value = int(user_input)
            if 1 <= value <= 5:
                return value
            else:
                print("Hiba: 1 és 5 közötti számot adj meg!")

        except ValueError:
            print("Hiba: számot kell megadni!")