import os
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

try:
    AUSTRALIAN_OPEN_DIR = 'data/australian_open_images/'
    TRAINING_AUSTRALIAN_DIR = 'data/tennis-data/training/australian/'
    TESTING_AUSTRALIAN_DIR = 'data/tennis-data/testing/australian/'

    US_OPEN_DIR = 'data/us_open_images/'
    TRAINING_US_DIR = 'data/tennis-data/training/us/'
    TESTING_US_DIR = 'data/tennis-data/testing/us/'

    FRENCH_OPEN_DIR = 'data/french_open_images/'
    TRAINING_FRENCH_DIR = 'data/tennis-data/training/french/'
    TESTING_FRENCH_DIR = 'data/tennis-data/testing/french/'

    WIMBLEDON_DIR = 'data/wimbledon_images/'
    TRAINING_WIMBLEDON_DIR = 'data/tennis-data/training/wimbledon/'
    TESTING_WIMBLEDON_DIR = 'data/tennis-data/testing/wimbledon/'

    file_paths = [
        AUSTRALIAN_OPEN_DIR,
        TRAINING_AUSTRALIAN_DIR,
        TESTING_AUSTRALIAN_DIR,
        US_OPEN_DIR,
        TRAINING_US_DIR,
        TESTING_US_DIR,
        FRENCH_OPEN_DIR,
        TRAINING_FRENCH_DIR,
        TESTING_FRENCH_DIR,
        WIMBLEDON_DIR,
        TRAINING_WIMBLEDON_DIR,
        TESTING_WIMBLEDON_DIR
    ]

    for file_path in file_paths:
        os.makedirs(file_path, exist_ok=True)

except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE=.90):
    list_of_all_images_in_source_directory = os.listdir(SOURCE)
    list_of_all_images_in_source_directory = random.sample(list_of_all_images_in_source_directory,
                                                           int(len(list_of_all_images_in_source_directory)))

    training_images = random.sample(list_of_all_images_in_source_directory,
                                    int(len(list_of_all_images_in_source_directory) * SPLIT_SIZE))

    testing_images = list(set(list_of_all_images_in_source_directory) - set(training_images))

    for image in training_images:
        if os.path.getsize(os.path.join(SOURCE, image)) > 0:
            copyfile(os.path.join(SOURCE, image), os.path.join(TRAINING, image))

    for image in testing_images:
        if os.path.getsize(os.path.join(SOURCE, image)) > 0:
            copyfile(os.path.join(SOURCE, image), os.path.join(TESTING, image))


# split_data(AUSTRALIAN_OPEN_DIR, TRAINING_AUSTRALIAN_DIR, TESTING_AUSTRALIAN_DIR)
# split_data(US_OPEN_DIR, TRAINING_US_DIR, TESTING_US_DIR)
# split_data(FRENCH_OPEN_DIR, TRAINING_FRENCH_DIR, TESTING_FRENCH_DIR)
# split_data(WIMBLEDON_DIR, TRAINING_WIMBLEDON_DIR, TESTING_WIMBLEDON_DIR)

def view_single_image(path, filename):
    img = mpimg.imread(os.path.join(path, filename))
    plt.imshow(img)
    plt.show()


view_single_image(TRAINING_US_DIR, os.listdir(TRAINING_US_DIR)[3])
