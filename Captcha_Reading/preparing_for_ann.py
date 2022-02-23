import numpy as np


def scale_to_range(image):
    return image/255


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))

    return ready_for_ann


def convert_output(alphabet):
    return np.eye(len(alphabet))
