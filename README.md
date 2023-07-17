# Handwritten_digit_Recognition

1. Imports necessary libraries: os, cv2, numpy, tensorflow, and matplotlib.pyplot.

2. Defines the variable train_new_model to decide whether to train a new model or load an existing one.

3. If train_new_model is True:
   a. Loads the MNIST dataset using tf.keras.datasets.mnist.
   b. Normalizes the training and testing data using tf.keras.utils.normalize.
   c. Creates a neural network model using tf.keras.models.Sequential.
   d. Adds a flattened input layer, two dense hidden layers, and a dense output layer with softmax activation.
   e. Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.
   f. Trains the model on the training data for 3 epochs.
   g. Evaluates the model on the testing data and prints the loss and accuracy.
   h. Saves the model using model.save().

4. If train_new_model is False:
   a. Loads the pre-trained model using tf.keras.models.load_model().

5. Loads custom images from the digits directory, predicts the digit using the model, and displays the image and predicted digit using plt.imshow() and plt.show().
   The code iterates through image files named as digit{}.png where {} is the image number starting from 1.
   The images are read using cv2.imread() and preprocessed by inverting the colors and converting to a numpy array.
   The model predicts the digit using model.predict() and prints the predicted digit.

6. Handles errors that may occur when reading the custom images and continues with the next image.

   To use this code, make sure you have the following:

  * The MNIST dataset accessible through tf.keras.datasets.mnist.
  * Custom digit images in the digits directory, named as digit{}.png where {} is the image number starting from 1.
  * Required libraries installed: opencv-python, numpy, tensorflow, and matplotlib.


##  You can run the code as-is or modify it based on your specific requirements and dataset paths.
