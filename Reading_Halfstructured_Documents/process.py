# import libraries here
import datetime
import cv2
from PIL import Image
import sys
import pyocr
import pyocr.builders
import matplotlib
import matplotlib.pyplot as plt
import image_preprocessing
from scipy import ndimage
import re

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)

tool = tools[0]
print("Koristimo backend: %s" % (tool.get_name()))
lang = 'eng'


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """

    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(image_path: str) -> Person:
    """
    Procedura prima putanju do slike sa koje treba ocitati vrednosti, a vraca objekat tipa Person, koji predstavlja osobu sa licnog dokumenta.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param image_path: <str> Putanja do slike za obradu
    :return: Objekat tipa "Person", gde su svi atributi setovani na izvucene vrednosti
    """
    person = Person('test', datetime.date.today(), 'test', 'test', 'test')

    # TODO - Prepoznati sve neophodne vrednosti o osobi sa slike. Vrednosti su: Name, Date of Birth, Job,
    #       Social Security Number, Company Name

    """ Pretprocesiranje za rotiranje """

    image = image_preprocessing.load_image(image_path)
    image_hsv = image_preprocessing.image_hsv(image)

    image_bin = binary_image(image_hsv)
    image_rect, contours_array = image_preprocessing.select_barcode(image.copy(), image_bin)

    if len(contours_array) == 0 or len(contours_array) > 1:
        return person

    image_rotated = ndimage.rotate(image, contours_array[0][1][2], reshape=False, cval=160)

    # image_preprocessing.display_image(image_rect, color=True)
    # image_preprocessing.display_image(image_rotated, color=True)
    # image_preprocessing.display_image(image_bin)

    print(image_path)

    """ Klasifikacija """

    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(image_rotated), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )

    founded_company = None
    for i, line in enumerate(line_and_word_boxes):
        if 'Apple' in line.content:
            founded_company = 'Apple'
            break
        elif 'IBM' in line.content:
            founded_company = 'IBM'
            break

        print()

    if founded_company is None:
        person.company = 'Google'
    else:
        person.company = founded_company

    """ Pretprocesiranje za citanje teksta (isecanje slike) """

    image_bin = binary_image2(image_preprocessing.image_hsv(image_rotated))
    image_rect, contours_array, image_crop = image_preprocessing.select_barcode_rotated_image(image_rotated.copy(),
                                                                                              image_bin, person.company)

    if len(contours_array) == 0 or len(contours_array) > 1:
        return person

    # image_preprocessing.display_image(image_rect, color=True)
    # image_preprocessing.display_image(image_crop, color=True)

    """ Citanje teksta """

    print("Procitan tekst: *****************************")
    line_and_word_boxes = tool.image_to_string(
        Image.fromarray(image_crop), lang=lang,
        builder=pyocr.builders.LineBoxBuilder(tesseract_layout=12)
    )

    for i, line in enumerate(line_and_word_boxes):
        print('line %d' % i)
        print(line.content, line.position)

        print(re.search("[0-9]{1,2}, [A-Z][a-z]{2} [0-9]{4}", line.content))

        if (len(line.content) > 3 and line.content in 'Manager') or 'Manager' in line.content:
            person.job = 'Manager'
            continue
        elif len(line.content) > 3 and line.content in 'Scrum Master' or 'Scrum Master' in line.content:
            person.job = 'Scrum Master'
            continue
        elif len(line.content) > 3 and line.content in 'Team Lead' or 'Team Lead' in line.content:
            person.job = 'Team Lead'
            continue
        elif len(line.content) > 3 and line.content in 'Human Resources' or 'Human Resources' in line.content:
            person.job = 'Human Resources'
            continue
        elif len(line.content) > 3 and line.content in 'Software Engineer' or 'Software Engineer' in line.content:
            person.job = 'Software Engineer'
            continue
        elif re.search("([A-Za-z]+\\.?\\s){1,2}[A-Za-z]", line.content) is not None:
            if person.name == 'test':
                person.name = re.search("([A-Za-z]+\\.?\\s){1,2}[A-Za-z]+", line.content).group()
                continue
        elif re.search("[0-9]{3}-[0-9]{2}-[0-9]{4}", line.content) is not None:
            person.ssn = re.search("[0-9]{3}-[0-9]{2}-[0-9]{4}", line.content).group()
            continue
        elif re.search("[0-9]{1,2}, [A-Z][a-z]{2} [0-9]{4}", line.content) is not None:
            if person.date_of_birth == datetime.date.today():
                date = re.search("[0-9]{1,2}, [A-Z][a-z]{2} [0-9]{4}", line.content).group()
                try:
                    person.date_of_birth = datetime.datetime.strptime(date, "%d, %b %Y").date()
                    continue
                except:
                    continue

        print()

    print(person.name, person.job, person.company, person.ssn, person.date_of_birth)
    # image_preprocessing.display_image(image_crop, color=True)

    print("--------------------------------------------------------------------------------")

    return person


def binary_image(image_hsv):
    image_bin = image_preprocessing.image_binary_by_color(image_hsv, [0, 0, 0], [180, 255, 0])
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)));
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)));
    return image_bin


def binary_image2(image_hsv):
    image_bin = image_preprocessing.image_binary_by_color(image_hsv, [0, 0, 0], [180, 255, 0])
    image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)));
    return image_bin
