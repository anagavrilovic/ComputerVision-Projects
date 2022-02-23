import matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


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


def image_binary_by_letter_color(img, lower_color, upper_color):
    # inRange() function
    # https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_bin = cv2.inRange(img, np.array(lower_color), np.array(upper_color))
    return img_bin


def invert(image):
    return 255-image


def display_image(image, color=False):
    plt.figure(figsize=(17, 9))
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region, (30, 30), interpolation=cv2.INTER_NEAREST)


def resize_image(image, width_scale, height_scale):
    width = int(image.shape[1] * width_scale)
    height = int(image.shape[0] * height_scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def filter_merged_regions(merged_regions):
    filtered_regions = []

    if len(merged_regions) != 0:
        max_width = merged_regions[0][1][2]
        max_height = merged_regions[0][1][3]

        for region in merged_regions:
            w = region[1][2]
            h = region[1][3]
            if w > max_width:
                max_width = w
            if h > max_height:
                max_height = h

        for region in merged_regions:
            if region[1][2] > max_width / 10 and region[1][3] > max_height / 3:
                filtered_regions.append(region)

    return filtered_regions


def select_roi(image_orig, image_bin, find_distances=True, hard=False):
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_height, max_width = find_biggest_contour(contours)

    regions_array = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if max_height/15 < h <= max_height and max_width/15 < w <= max_width:
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])

    regions_array = sorted(regions_array, key=lambda item: item[1][0])

    if not hard:
        merged_regions = merge_letters_with_hooks_and_dots(image_bin, regions_array)
    else:
        merged_regions = merge_letters_with_hooks_and_dots_hard(image_bin, regions_array)

    merged_regions = filter_merged_regions(merged_regions)

    for region in merged_regions:
        cv2.rectangle(image_orig, (region[1][0], region[1][1]), (region[1][0] + region[1][2], region[1][1] + region[1][3]), (0, 255, 0), 2)

    sorted_regions = [region[0] for region in merged_regions]
    sorted_rectangles = [region[1] for region in merged_regions]

    if find_distances:
        region_distances = find_distances_between_letters(sorted_rectangles)
    else:
        region_distances = []

    return image_orig, sorted_regions, region_distances


def find_biggest_contour(contours):
    max_width = 0
    max_height = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > max_width:
            max_width = w
        if h > max_height:
            max_height = h
    return max_height, max_width


def merge_letters_with_hooks_and_dots(image_bin, regions_array):
    merged_regions = []
    for idx in range(len(regions_array)):
        current_region = regions_array[idx]
        if idx == 0:
            merged_regions.append(current_region)
        else:
            previous_region = regions_array[idx - 1]
            if (previous_region[1][0] + previous_region[1][2] > current_region[1][0] + current_region[1][2] / 2 and
                    current_region[1][2] < previous_region[1][2] and
                    current_region[1][3] < previous_region[1][3] / 2) or \
                (previous_region[1][0] + previous_region[1][2] > current_region[1][0] + current_region[1][2] / 2 and
                    previous_region[1][2] < current_region[1][2] and
                    previous_region[1][3] < current_region[1][3] / 2):

                merged_regions.pop()
                region_x = min(previous_region[1][0], current_region[1][0])
                region_y = min(previous_region[1][1], current_region[1][1])
                region_x_max = max(previous_region[1][0] + previous_region[1][2], current_region[1][0] + current_region[1][2])
                region_y_max = max(previous_region[1][1] + previous_region[1][3], current_region[1][1] + current_region[1][3])
                region_w = region_x_max - region_x
                region_h = region_y_max - region_y

                region = image_bin[region_y:region_y + region_h + 1, region_x:region_x + region_w + 1]
                merged_regions.append([resize_region(region), (region_x, region_y, region_w, region_h)])
            else:
                merged_regions.append(current_region)
    return merged_regions


def merge_letters_with_hooks_and_dots_hard(image_bin, regions_array):
    merged_regions = []
    for idx in range(len(regions_array)):
        current_region = regions_array[idx]
        if idx == 0:
            merged_regions.append(current_region)
        else:
            previous_region = regions_array[idx - 1]
            if (previous_region[1][0] + previous_region[1][2] > current_region[1][0] + current_region[1][2] / 2 and
                    current_region[1][2] < previous_region[1][2] and
                    current_region[1][3] < previous_region[1][3] / 2 and
                    previous_region[1][1] - (current_region[1][1] + current_region[1][3]) < current_region[1][3]*1.5 and
                    current_region[1][1] < previous_region[1][1]) or \
                (previous_region[1][0] + previous_region[1][2] > current_region[1][0] + current_region[1][2] / 2 and
                    previous_region[1][2] < current_region[1][2] and
                    previous_region[1][3] < current_region[1][3] / 2 and
                    current_region[1][1] - (previous_region[1][1] + previous_region[1][3]) < current_region[1][3]*1.5 and
                    current_region[1][1] < previous_region[1][1]):

                merged_regions.pop()
                region_x = min(previous_region[1][0], current_region[1][0])
                region_y = min(previous_region[1][1], current_region[1][1])
                region_x_max = max(previous_region[1][0] + previous_region[1][2], current_region[1][0] + current_region[1][2])
                region_y_max = max(previous_region[1][1] + previous_region[1][3], current_region[1][1] + current_region[1][3])
                region_w = region_x_max - region_x
                region_h = region_y_max - region_y

                region = image_bin[region_y:region_y + region_h + 1, region_x:region_x + region_w + 1]
                merged_regions.append([resize_region(region), (region_x, region_y, region_w, region_h)])
            else:
                merged_regions.append(current_region)
    return merged_regions


def find_distances_between_letters(sorted_rectangles):
    region_distances = []
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])  # X_next - (X_current + W_current)
        region_distances.append(distance)
    return region_distances
