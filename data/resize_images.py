from PIL import Image


def resize_image(image, width, height):
    img = Image.open(image)
    return img.resize((width, height))