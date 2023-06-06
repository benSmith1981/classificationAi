# Import the necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess the data: Before using the pixel values of the images as input to the neural network, 
# they are normalized to be in the range [0, 1] by dividing each pixel value by 255. This can help the neural network learn more effectively.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model: This creates a Sequential model using Keras. This is a type of neural network where the layers are stacked sequentially. 
# The Flatten layer reshapes each 28x28 image into a 1D array of 784 elements. 
# The Dense layer with 128 neurons is the hidden layer, and the relu (Rectified Linear Unit) activation function is used. 
# The last Dense layer is the output layer, it has 10 neurons (one for each class), and the softmax activation function is used to output a probability distribution over the 10 classes.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model: This configures the learning process of the model. The adam optimizer is used, which is an efficient variant of stochastic gradient descent. 
# The sparse_categorical_crossentropy loss function is used, 
# which is suitable for multiclass classification problems like this one. The accuracy of the predictions is also computed during training and evaluation.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model: This feeds the training images and labels to the model, and runs the optimization process. 
# This process is run for 10 epochs, meaning the entire dataset is passed through the model 10 times.
model.fit(train_images, train_labels, epochs=100)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# After training your model, save it
model.save('fashion_mnist_model.h5')
