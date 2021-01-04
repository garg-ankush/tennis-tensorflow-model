import os
import random
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from data.resize_images import resize_image
from PIL import Image

# Create directories for tensor flow model to get images from
# Create Training and Testing directories for each category

image_width = 128
image_height = 128


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
    # Get list of all images in the grand slam folder
    list_of_all_images_in_source_directory = os.listdir(SOURCE)
    # Shuffle all images
    list_of_all_images_in_source_directory = random.sample(list_of_all_images_in_source_directory,
                                                           int(len(list_of_all_images_in_source_directory)))

    # Take 90% of images as training images
    training_images = random.sample(list_of_all_images_in_source_directory,
                                    int(len(list_of_all_images_in_source_directory) * SPLIT_SIZE))

    # Take the remaining images as testing images
    testing_images = list(set(list_of_all_images_in_source_directory) - set(training_images))

    # Loop to copy training images to their designated path
    for image in training_images:
        image_source_path = os.path.join(SOURCE, image)
        # Only move images if their size is greater than 0
        if os.path.getsize(image_source_path) > 0:
            img = resize_image(image_source_path, image_width, image_height)
            img.save(os.path.join(TRAINING, image))

    # Loop to copy testing images to their designated path
    for image in testing_images:
        image_source_path = os.path.join(SOURCE, image)
        # Only move images if their size is greater than 0
        if os.path.getsize(image_source_path) > 0:
            img = resize_image(image_source_path, image_width, image_height)
            img.save(os.path.join(TESTING, image))


# Call split data on each of the grand slams
split_data(AUSTRALIAN_OPEN_DIR, TRAINING_AUSTRALIAN_DIR, TESTING_AUSTRALIAN_DIR)
split_data(US_OPEN_DIR, TRAINING_US_DIR, TESTING_US_DIR)
split_data(FRENCH_OPEN_DIR, TRAINING_FRENCH_DIR, TESTING_FRENCH_DIR)
split_data(WIMBLEDON_DIR, TRAINING_WIMBLEDON_DIR, TESTING_WIMBLEDON_DIR)


# Test single image to confirm that files were copied as intended
def view_single_image(path, filename):
    img = mpimg.imread(os.path.join(path, filename))
    plt.imshow(img)
    plt.show()


# View single file
# view_single_image(TRAINING_US_DIR, os.listdir(TRAINING_US_DIR)[0])
