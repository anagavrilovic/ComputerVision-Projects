import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def blur_image(image):
    return cv2.GaussianBlur(image, (3, 3), 0)


def image_otsu_binary(image_gs):
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
    return image_bin


def image_ada_binary(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 55, 25)
    return image_bin


def image_thresh_binary(image_gs, thresh):
    ret, image_bin = cv2.threshold(image_gs, thresh, 255, cv2.THRESH_BINARY)
    return image_bin


def image_binary_by_color(img, lower_color, upper_color):
    # inRange() function
    # https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_bin = cv2.inRange(img, np.array(lower_color), np.array(upper_color))
    return img_bin


def invert(image):
    return 255-image


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def display_image(image, color=False):
    # plt.figure(figsize=(17, 9))
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def resize_image(image, width_scale, height_scale):
    width = int(image.shape[1] * width_scale)
    height = int(image.shape[0] * height_scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def fix_angle(angle):
    if angle < -45:
        return 90 + angle
    else:
        return angle


def select_barcode(image_orig, image_bin):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_array = []
    filtered_contours = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        if ((image_bin.shape[1] / 60 < size[0] < image_bin.shape[1] / 20 and image_bin.shape[0] / 2 > size[1] > image_bin.shape[0] / 10) or \
            (image_bin.shape[0]/60 < size[1] < image_bin.shape[0]/15 and image_bin.shape[1]/3 > size[0] > image_bin.shape[1]/10)) and \
                (size[1] < size[0]/5 or size[0] < size[1]/5) and \
                20 < center[1] < image_bin.shape[0] - 20:
            filtered_contours.append(contour)
            angle = fix_angle(angle)
            contours_array.append([contour, (center, size, angle)])

    cv2.drawContours(image_orig, np.array(filtered_contours), -1, (255, 0, 0), 2)

    return image_orig, contours_array


def select_barcode_rotated_image(image_orig, image_bin, company):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_array = []
    # filtered_contours = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        if ((image_bin.shape[1] / 70 < size[0] < image_bin.shape[1] / 20 and image_bin.shape[0] / 2 > size[1] > image_bin.shape[0] / 10) or \
            (image_bin.shape[0]/70 < size[1] < image_bin.shape[0]/15 and image_bin.shape[1]/3 > size[0] > image_bin.shape[1]/10)) and \
                (size[1] < size[0]/4 or size[0] < size[1]/4) and \
                50 < center[1] < image_bin.shape[0] - 50 and \
                (-3 < angle < 3 or -93 < angle < -87 or 87 < angle < 93):
            # filtered_contours.append(contour)
            contours_array.append([contour, (center, size, angle)])

    image_crop = None
    if len(contours_array) == 1:
        image_crop = crop_id_card(image_orig, contours_array[0][1], company)

    # cv2.drawContours(image_orig, np.array(filtered_contours), -1, (255, 0, 0), 2)

    return image_orig, contours_array, image_crop


def crop_id_card(image, contour, company):
    ((x, y), (w, h), angle) = contour
    x = int(x)
    y = int(y)
    h = int(h)
    w = int(w)

    if company == 'Apple':
        if h < w:
            q = h
            h = w
            w = q

        a = int(y - h // 1.9)
        if a < 0:
            a = 0
        b = int(y + h // 1.9)
        if b > image.shape[0]:
            b = image.shape[0]
        c = int(x + w * 1.5)
        if c < 0:
            c = 0
        d = int(x + h * 1.3)
        if d > image.shape[1]:
            d = image.shape[1]

        image = image[a:b, c:d]
    elif company == 'IBM':
        if w < h:
            q = h
            h = w
            w = q

        a = int(y - h)
        if a < 0:
            a = 0
        b = int(y + w // 1.6)
        if b > image.shape[0]:
            b = image.shape[0]
        c = int(x - w // 1.8)
        if c < 0:
            c = 0
        d = int(x + w // 1.9)
        if d > image.shape[1]:
            d = image.shape[1]
        image = image[a:b, c:d]
    elif company == 'Google':
        if w < h:
            q = h
            h = w
            w = q

        a = int(y - h * 0.7)
        if a < 0:
            a = 0
        b = int(y + w)
        if b > image.shape[0]:
            b = image.shape[0]
        c = int(x - w * 2.6)
        if c < 0:
            c = 0
        d = int(x - w // 1.8)
        if d > image.shape[1]:
            d = image.shape[1]
        print(a, b, c, d)
        image = image[a:b, c:d]

    return image
