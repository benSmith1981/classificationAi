from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the model
loaded_model = keras.models.load_model('fashion_mnist_model.h5')


# Define the class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/teachable')
def teachable():
    return render_template('teachable.html')

# Make a prediction
def predict_image(image):
    image = image.reshape(1, 28, 28)  # reshape the image to the input shape of the model
    prediction = loaded_model.predict(image)
    predicted_class = np.argmax(prediction)
    return class_names[predicted_class]

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # get the file
    image = Image.open(file).convert('L')  # convert image to grayscale
    image = image.resize((28, 28))  # resize image to 28x28
    image = np.array(image) / 255.0  # normalize the pixel values
    prediction = predict_image(image)
    print(prediction)
    return render_template('prediction.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)