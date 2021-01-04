import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np


# Initialize sequential model with 3 Conv2D layers and 3 MaxPool2D layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

print(model.summary())
# Compile model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Rescale images and initialize ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1/255.0)
validation_datagen = ImageDataGenerator(rescale=1/255.0)

# Read images from training directory and initialize train_datagenerator
TRAINING_DIR = 'data/tennis-data/training'
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(128, 128),
    batch_size=10,
    class_mode='categorical'
)

# Read images from validation directory and initialize validation_datagenerator
VALIDATION_DIR = 'data/tennis-data/testing'
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(128, 128),
    batch_size=10,
    class_mode='categorical'
)

# Fit model for 3 epochs
history = model.fit(
    train_generator,
    epochs=3,
    verbose=1,
    validation_data=validation_generator
)

# Get model metrics from history object
acc = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
# Plot Accuracy
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', 'Validation Accuracy')

# Plot loss
plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.show()


# Function to test a single image
def test_single_image(model_name, filename):
    testing_image = image.load_img(filename)
    img = image.img_to_array(testing_image)
    img = np.expand_dims(img, axis=0)

    classes = model_name.predict(img, batch_size=10)
    return classes


testing_image_path = '/Users/ankushgarg/Desktop/tennis-tf/data/tennis-data/testing/australian/australian_open_img_9.jpg'
test_single_image(model, testing_image_path)
