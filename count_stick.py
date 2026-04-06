# import moduls
import cv2

import add_noise
import show_thinning
import analyse_thinning
import analyse_img

# source_img = 'pic/palcika1.jpg'
# img_source = 'pic/palcika2.jpg'
# img_source = 'pic/palcika3.png'
img_source = 'pic/palcika4.webp'

noise_level = add_noise.get_noise_level()
print("Zaj szint:", noise_level)

# Kép beolvasása fájlból
im_src = cv2.imread(img_source, cv2.IMREAD_COLOR)
cv2.imshow('01 - source image color', im_src)
cv2.waitKey(0)

# create_img_with_noise
img_noisy = add_noise.create_img_with_noise(im_src, noise_level)
cv2.imshow('02 - source image color - with noise', img_noisy)
cv2.waitKey(0)

# Szürkeárnyalatos konverzió
img_gray = cv2.cvtColor(img_noisy, cv2.COLOR_BGR2GRAY)
cv2.imshow('03 - source image - with noise - in gray', img_gray)
cv2.waitKey(0)

# GaussianBlur
img_filtered = cv2.GaussianBlur(img_gray, (21, 21), sigmaX=3.0, sigmaY=3.0)
cv2.imshow('04 - noise image in gray - GaussianBlur', img_filtered)
cv2.waitKey(0)

# Otsu módszerrel történő globális küszöbölés
# cv2.THRESH_BINARY: fehér pálcikák, fekete háttér
_, img_otsu = cv2.threshold(img_filtered, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('05 - threshold OTSU', img_otsu)
cv2.waitKey(0)

# invert img - fehér hátterű kép esetén invertálni kell a képet
inverted, img_otsu = analyse_img.invert_img(img_otsu)
if inverted:
    cv2.imshow('05 - inverted threshold OTSU', img_otsu)
    cv2.waitKey(0)

# Thinning (skeletonizáció)
img_thin = cv2.ximgproc.thinning(img_otsu, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
cv2.imshow('06 - thinning', img_thin)
cv2.waitKey(0)

# Lépésenként a vonal csoportok megjelenítése
show_thinning.show_sticks_group(img_otsu, img_thin)

# Összes vonal csoport megjelenítése
img_overlay = show_thinning.show_sticks_all_groups(img_otsu, img_thin)
cv2.imshow('07 - all thinning groups - img_otsu + thinning', img_overlay)
cv2.waitKey(0)

# végpontok keresése
img_end_point = analyse_thinning.get_overlay_with_end_point_mask(img_otsu, img_thin)
cv2.imshow('08 - end points', img_end_point)
cv2.waitKey(0)

# pálcikák számának meghatározása
num_endpoints = analyse_thinning.get_number_of_sticks(img_thin)
print("Pálcikák száma:", num_endpoints)

# Összes ablak bezárása
cv2.destroyAllWindows()