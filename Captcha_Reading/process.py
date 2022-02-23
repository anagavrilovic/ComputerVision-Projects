import image_preprocessing
import preparing_for_ann
import ann_functions
import text_postprocessing

import numpy as np
import cv2

# Sklearn biblioteka sa implementiranim K-means algoritmom
from sklearn import datasets
from sklearn.cluster import KMeans


alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž',
            'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']


def preprocess_train_images(train_image_paths):
    input_letter_images = []

    for image_path in train_image_paths:
        image = image_preprocessing.load_image(image_path)

        image_bin = image_preprocessing.invert(image_preprocessing.image_otsu_binary(image_preprocessing.image_gray(image)))
        image_bin = image_preprocessing.erode(image_bin)
        image_bin = image_preprocessing.dilate(image_bin)
        # image_preprocessing.display_image(image_bin, color=False)

        image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin, find_distances=False)
        input_letter_images = [*input_letter_images, *sorted_regions]  # concat elements of two lists
        print(len(input_letter_images))
        # image_preprocessing.display_image(image_orig, color=True)

    return input_letter_images


def preprocess_validation_image(validation_image_path):
    image = image_preprocessing.load_image(validation_image_path)

    if image.shape[1] < 4000:
        image = image_preprocessing.resize_image(image, 2, 1.2)

    # image = image_preprocessing.resize_image(image, 1.7, 2.2)
    image_blur = image_preprocessing.blur_image(image)
    image_bin = image_preprocessing.image_binary_by_letter_color(image_blur, [0, 0, 200], [180, 255, 255])

    if cv2.countNonZero(image_bin) > image.shape[0] * image.shape[1] - cv2.countNonZero(image_bin):
        # image = image_preprocessing.resize_image(image, 0.6, 0.45)
        image_gray = image_preprocessing.blur_image(image_preprocessing.image_gray(image))
        image_bin = image_preprocessing.invert(image_preprocessing.image_otsu_binary(image_gray))
        image_bin = image_preprocessing.erode(image_bin)
        image_bin = image_preprocessing.dilate(image_bin)

    image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin,
                                                                                  find_distances=True, hard=True)
    # image_bin = image_preprocessing.erode(image_bin)
    # image_bin = image_preprocessing.dilate(image_bin)
    # image_preprocessing.display_image(image_orig, color=True)
    # image_preprocessing.display_image(image_bin, color=False)

    # ponovo svetla slova
    if len(sorted_regions) < 10 or (image.shape[0]*image.shape[1] - cv2.countNonZero(image_bin)) > image.shape[0]*image.shape[1]*0.99:
        image = image_preprocessing.resize_image(image, 1.7, 2.2)
        image_blur = image_preprocessing.blur_image(image)
        image_bin = image_preprocessing.image_binary_by_letter_color(image_blur, [0, 0, 210], [180, 255, 255])
        image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin, find_distances=True, hard=True)

        # ljubicasta/roze/plava slova
        if len(sorted_regions) < 10 or (image.shape[0] * image.shape[1] - cv2.countNonZero(image_bin)) > image.shape[0] *image.shape[1] * 0.99:
            image_blur = image_preprocessing.blur_image(image)
            image_bin = image_preprocessing.image_binary_by_letter_color(image_blur, [70, 20, 115], [179, 255, 255])
            image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin, find_distances=True, hard=True)

            # narandzasta slova
            if len(sorted_regions) < 10 or (image.shape[0] * image.shape[1] - cv2.countNonZero(image_bin)) > image.shape[0] *image.shape[1] * 0.99:
                image_blur = image_preprocessing.blur_image(image)
                image_bin = image_preprocessing.image_binary_by_letter_color(image_blur, [19, 45, 180], [40, 225, 255])
                image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin, find_distances=True, hard=True)

                # crvena slova
                if len(sorted_regions) < 10 or (image.shape[0] * image.shape[1] - cv2.countNonZero(image_bin)) > image.shape[0] * image.shape[1] * 0.99:
                    image_blur = image_preprocessing.blur_image(image)
                    image_bin = image_preprocessing.image_binary_by_letter_color(image_blur, [0, 20, 180], [20, 120, 240])
                    image_orig, sorted_regions, region_distances = image_preprocessing.select_roi(image.copy(), image_bin, find_distances=True, hard=True)

    # image_preprocessing.display_image(image_bin, color=False)

    # image_preprocessing.display_image(image_orig, color=True)
    # image_preprocessing.display_image(image_bin, color=False)

    return sorted_regions, region_distances


def train_or_load_character_recognition_model(train_image_paths):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta)

    Procedura treba da istrenira model i da ga sacuva pod proizvoljnim nazivom. Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran

    :param train_image_paths: putanje do fotografija alfabeta
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati ako je vec istreniran

    input_letters = preprocess_train_images(train_image_paths)

    model = ann_functions.load_trained_ann()

    if model is None:
        inputs = preparing_for_ann.prepare_for_ann(input_letters)
        outputs = preparing_for_ann.convert_output(alphabet)

        print("Treniranje modela zapoceto.")
        model = ann_functions.create_ann()
        model = ann_functions.train_ann(model, inputs, outputs)
        print("Treniranje modela zavrseno.")
        ann_functions.serialize_ann(model)

    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    extracted_text = ""
    # TODO - Izvuci tekst sa ulazne fotografije i vratiti ga kao string

    # '''
    sorted_regions, region_distances = preprocess_validation_image(image_path)

    region_distances = np.array(region_distances).reshape(len(region_distances), 1)
    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    if len(region_distances) > 3:
        k_means.fit(region_distances)

        inputs = preparing_for_ann.prepare_for_ann(sorted_regions)
        results = trained_model.predict(np.array(inputs, np.float32))

        extracted_text = text_postprocessing.display_result(results, alphabet, k_means)

    print("-----------------------------------------------------")
    print(extracted_text)
    extracted_text = text_postprocessing.fuzzy_wuzzy(extracted_text, vocabulary)
    print(extracted_text)
    # '''

    return extracted_text
