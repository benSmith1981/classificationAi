from flask import Flask, request
from PIL import Image
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the model
loaded_model = keras.models.load_model('fashion_mnist_model.h5')


# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Make a prediction
def predict_image(image):
    image = image.reshape(1, 28, 28)  # reshape the image to the input shape of the model
    prediction = loaded_model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]


@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion MNIST Classifier</title>
    </head>
    <body>
        <h1>Fashion MNIST Classifier</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Classify Image">
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # get the file
    image = Image.open(file).convert('L')  # convert image to grayscale
    image = image.resize((28, 28))  # resize image to 28x28
    image = np.array(image) / 255.0  # normalize the pixel values
    prediction = predict_image(image)
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion MNIST Classifier</title>
    </head>
    <body>
        <h1>Fashion MNIST Classifier</h1>
        <p>Predicted class: {}</p>
        <a href="/">Try again</a>
    </body>
    </html>
    '''.format(prediction)

if __name__ == "__main__":
    app.run(debug=True)