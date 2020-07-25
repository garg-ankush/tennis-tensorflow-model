import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1/255.0)
validation_datagen = ImageDataGenerator(rescale=1/255.0)

TRAINING_DIR = 'data/tennis-data/training'
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

VALIDATION_DIR = 'data/tennis-data/testing'
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical'
)

history = model.fit_generator(
    train_generator,
    epochs=1,
    verbose=1,
    validation_data=validation_generator
)

# acc = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'r', 'Training Accuracy')
# plt.plot(epochs, val_accuracy, 'b', 'Validation Accuracy')
#
# plt.plot(epochs, loss, 'r', 'Training Loss')
# plt.plot(epochs, val_loss, 'b', 'Validation Loss')
# plt.show()

testing_image_path = '/Users/ankushgarg/Desktop/wimbledon.jpeg'
testing_image = image.load_img(testing_image_path, target_size=(150, 150, 3))
x = image.img_to_array(testing_image)
x = np.expand_dims(x, axis=0)

classes = model.predict(x, batch_size=10)
print(classes)