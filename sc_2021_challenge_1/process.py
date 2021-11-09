# import libraries here
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy import signal
from scipy.cluster.vq import vq, kmeans, whiten


def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def show_binary_image(img):
    plt.imshow(img, 'gray')
    plt.show()


def blur_image(img):
    (height, width, dim) = img.shape

    if width > 1000 and height > 1000:
        image_blur = cv2.GaussianBlur(img, (11, 11), 0)
    else:
        image_blur = cv2.GaussianBlur(img, (5, 5), 0)

    return image_blur


def binaryze_image(img):
    # inRange() function
    # https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html

    lower_gray = np.array([0, 0, 80])
    upper_gray = np.array([180, 60, 220])
    img_bin = cv2.inRange(img, lower_gray, upper_gray)
    return img_bin


def morph_fix_image(img_bin):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    img_bin = cv2.erode(img_bin, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bin = cv2.dilate(img_bin, kernel, iterations=1)
    img_bin = cv2.erode(img_bin, kernel, iterations=2)

    return img_bin


def find_region_number(img_bin):
    img, contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_contours(img, contours):
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()


def get_useful_contours(img, contours):
    car_contours = []

    (img_height, img_width, img_dim) = img.shape

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size

        if img_width > 1000 and img_height > 1000:
            if img_width / 15 < width < img_width / 2 and img_height / 15 < height < img_height / 2:
                car_contours.append(contour)
        else:
            if img_width / 60 < width < img_width / 2 and img_height / 60 < height < img_height / 2:
                car_contours.append(contour)

    return car_contours


def resize_image(img):
    # image resizing
    # https://learnopencv.com/image-resizing-with-opencv/

    width = int(img.shape[1] * 2.2)
    height = int(img.shape[0] * 2.2)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def watershed_algorithm(img, img_bin):
    """img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    plt.imshow(img)
    plt.show()"""


def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """
    car_count = 0
    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure

    img = load_image(image_path)

    img = blur_image(img)
    img_bin = binaryze_image(img)
    img_bin = 255 - img_bin
    img_bin = morph_fix_image(img_bin)

    # show_binary_image(img_bin)
    watershed_algorithm(img, img_bin)

    contours = find_region_number(img_bin)
    # draw_contours(img, contours)
    contours = get_useful_contours(img, contours)
    # draw_contours(img, contours)

    car_count = len(contours)

    return car_count

