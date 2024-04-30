import numpy as np
import mnist
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.optimizers import adam_v2

trainImages = mnist.train_images()
trainLabels = mnist.train_labels()
testImages = mnist.test_images()
testLabels = mnist.test_labels()

trainImages = (trainImages / 255) - 0.5
testImages = (testImages / 255) - 0.5

trainImages = np.expand_dims(trainImages, axis = 3)
testImages = np.expand_dims(testImages, axis = 3)

model = Sequential([
    Conv2D(8, 3, input_shape = (28, 28, 1), use_bias=False),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(10, activation="softmax")
])

model.compile(adam_v2.Adam(learning_rate=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    trainImages,
    to_categorical(trainLabels),
    batch_size=32,
    epochs=3,
    validation_data=(testImages, to_categorical(testLabels)),
)