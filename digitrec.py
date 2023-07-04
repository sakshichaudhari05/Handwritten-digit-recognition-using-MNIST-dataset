# Import gzip to unzip the data files
import gzip
# Import io to readt the files by byte
import io
# Import url open to download the files
from urllib.request import urlopen
# Import pathlib for file opener
from pathlib import Path
# Import numpy for array operations
import numpy as np
# Import accuracy_core got the the accuracy of the classification methords
from sklearn.metrics import accuracy_score
# Import KNN as an alternative classifier
from sklearn.neighbors import KNeighborsClassifier
# Import MLPClassifier model
from sklearn.neural_network import MLPClassifier
# Import Gaussian classifier
from sklearn.naive_bayes import GaussianNB
# Import SVC classifier
from sklearn.svm import SVC
# Import image operations libraries
from PIL import Image, ImageOps
# Import keras for custom neuronetwork classification
import keras as kr
# Import preprocessing for binaryarray conversion
import sklearn.preprocessing as pre
# Import keras layers for the neuronetwork
from keras.layers import Dense, Dropout
# Import argparse for system arguments
import argparse
# Import walk fro dir reading
from os import walk
# Import is dir for checking if a value is directory
from os.path import isdir
# Import imghdr to determine if a file is image
import imghdr
# Import joblib to allow the model to be saved into a file
from sklearn.externals import joblib
import sys


class DigitRecognition:

    def __init__(self, verbose: bool = False, modelName: str = 'keras', limit: bool = False, checkAccuracy: bool = True, saveModel: str = None, loadModel: str = None, loadTestData: bool = True):
        """
        Constructor of class.
        Loads the training and test files
        verbose: extra details are printed about the files
        modelName: the model to be used for classification
        limit: to limit the training and test dataset
        checkAccuracy: to check the accuracy of the machine learning method or not
        saveModel: Saves the model into a file for later use
        loadModel: Loads an already trained model
        loadTestData: to load the validation data into memory or not
        """
        self.verbose = verbose
        self.loadTestData = loadTestData
        self.modelName = modelName
        self.limit = limit
        self.checkAccuracy = checkAccuracy

        # Load the model from a file if it is set
        if loadModel != None and loadModel != "":
            # Load a previously trained model
            self.__loadModel(loadModel)
        else:
            # Load the files
            self.__loadRawFiles()
            # Read the files into arrays
            self.__readTrainingAndTestData()
            # Prepare the model and execute it
            self.__selectModelAndRun()
            # Save model to file
            self.__saveModel(saveModel)

    def recogniseNumberFromPicture(self, image):
        """
        Tries to predict a number from a image
        """
        # Convert image to array
        pa = self.imageAsArray(image)
        # Predict with keras
        if self.modelName == 'keras':
            print('Prediced: %d' % np.argmax(self.__predictWithPreviousModel(
                pa.reshape(1, 784))), " for image ", image)
        else:
            # Predict with other models
            print("Predicted: ", self.__predictWithPreviousModel(
                [pa]), " for image ", image)

    def imageAsArray(self, imageName: str):
        """
        Reads in a image from a file and return it as a 784 size pixel array consiting with numbers from 0 to 1
        """
        # Open the image and convert to gray scale and finally invert it so it maches with the training data set
        i = ImageOps.invert(Image.open(imageName).convert('L'))
        # Resize the image the same size as the training images
        i = i.resize((28, 28))
        # Convert to array, flaten it out and normalise it
        return np.asarray(i).reshape(784)/255

    def __saveModel(self, filename):
        """
        Saves the model into a file for later use if the file name is not empty
        """
        if filename != None and filename != "":
            joblib.dump(self.model, filename)

    def __loadModel(self, filename):
        """
        Loads a previously trained model
        """
        try:
            # Try to load the model
            self.model = joblib.load(filename)
            # Check model type
            if isinstance(self.model, kr.models.Sequential) == True:
                self.modelName = "keras"
            # Set up the model if it knn
            if isinstance(self.model, KNeighborsClassifier) == True:
                self.modelName = "knn"
            # Set up the model if it mlpc
            if isinstance(self.model, MLPClassifier) == True:
                self.modelName = "mlpc"
            # Set up the model if it gaussian
            if isinstance(self.model, GaussianNB) == True:
                self.modelName = "gaussian"
            # Set up the model if it svc
            if isinstance(self.model, SVC) == True:
                self.modelName = "svc"

            print("Using preloaded model: ", self.modelName)

        except FileNotFoundError:
            print(filename, "does not exist!")
            # Terminate
            sys.exit()

    def __loadRawFiles(self)->None:
        """
        Unzip the training and test data
        """
        self.train_images = self.__openOrDownloadAndOpenGzipFile(
            'data/train-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
        self.train_labels = self.__openOrDownloadAndOpenGzipFile(
            'data/train-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
        if self.loadTestData:
            self.test_images = self.__openOrDownloadAndOpenGzipFile(
                'data/t10k-images-idx3-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
            self.test_labels = self.__openOrDownloadAndOpenGzipFile(
                'data/t10k-labels-idx1-ubyte.gz', 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
        print("Data files loaded.")

    def __openOrDownloadAndOpenGzipFile(self, file_name: str, url: str)->str:
        """
        Checks if the given file exists, if it doesnt then downloads from the given url
        """
        if self.verbose:
            print('Checking %s' % file_name)

        file = Path(file_name)
        # Check if the file exist
        if(file.is_file() != True):
            print('Downloading %s' % url)
            # Download and read if not
            to_read = urlopen(url).read()
            # Save the file
            sfile = open(file_name, 'wb')
            sfile.write(to_read)
            sfile.close()
        else:
            # Read if it does
            to_read = file.read_bytes()
        # unzip the file
        with gzip.open(io.BytesIO(to_read), 'rb') as f:
            return f.read()

    def __readTrainingAndTestData(self):
        """
        Reads in both training and test data
        """
        print("\nReading training images and labels into arrays")
        self.train_images, self.train_labels, train_item_number = self.__readImagesAndLabels(
            self.train_images, self.train_labels)
        if self.loadTestData:
            print("\nReading test images and labels into arrays")
            self.test_images, self.test_labels, test_item_number = self.__readImagesAndLabels(
                self.test_images, self.test_labels)

    def __readImagesAndLabels(self, images_raw, labels_raw):
        """
        Reads the images and labels into arrays
        """
        # Get det images and label meta data
        images_number, columns_number, rows_number, label_number = self.__getImageAndLabelMetaData(
            images_raw, labels_raw)
        # Check if the numbers of images and labels match
        if images_number != label_number:
            raise Exception("The number of images and labels does not mach!")
        # Read the images and labels into arrays
        images, labels = self.__loadimagesAndLabelsToArrays(
            images_raw, images_number, columns_number, rows_number, 16, labels_raw, 8)

        return images, labels, images_number

    def __getImageAndLabelMetaData(self, images_raw, labels_raw):
        """
        Checks if the magic numbers are correct and reads in the first set of bites of the files.
        """
        if self.verbose:
            print("Meta data of images file:")

        # Confirm if the first four byteis 2051
        is_it_the_right_bytes = int.from_bytes(
            images_raw[0:4], byteorder='big') == 2051
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST images file")
        if self.verbose:
            print('Is the magic number correct: %r' % is_it_the_right_bytes)

        # Number of images should be from bytes 4 to 8 and should be read in big endian
        images_number = int.from_bytes(
            images_raw[4:8], byteorder='big')
        if self.verbose:
            print('Number of images: %d' % images_number)

        # Number of rows should be from 8 to 12
        rows_number = int.from_bytes(
            images_raw[8:12], byteorder='big')
        if self.verbose:
            print('Number of rows: %d' % rows_number)

        # Number of columns should be from 12 to 16
        columns_number = int.from_bytes(
            images_raw[12:16], byteorder='big')
        if self.verbose:
            print('Number of columns: %d' % columns_number)
        if self.verbose:
            print("Meta data of labels file:")

        # Confirm if the first four byteis 2049
        is_it_the_right_bytes = int.from_bytes(
            labels_raw[0:4], byteorder='big') == 2049
        # Throw exception if wrong file provided
        if is_it_the_right_bytes == False:
            raise Exception("The provided file is not MNIST labels file")
        if self.verbose:
            print('Is the magic number correct: %r' % is_it_the_right_bytes)

        # Number of images should be from bytes 4 to 8 and should be read in big endian
        label_number = int.from_bytes(
            labels_raw[4:8], byteorder='big')
        if self.verbose:
            print('Number of Labels: %d' % label_number)
        return images_number, columns_number, rows_number, label_number

    def __loadimagesAndLabelsToArrays(self, image_file_content, images_number: int, columns_number: int, rows_number: int, images_offset: int, label_file_content, labels_offset: int):
        """
        Loads a set of images and labels into two arrays.
        The number of images and labels has to match
        The method reads in each image flat as columns_number*rows_number.
        """
        images = np.frombuffer(image_file_content[images_offset:], dtype=np.uint8).reshape(
            images_number, columns_number*rows_number)/255
        labels = np.frombuffer(
            label_file_content[labels_offset:], dtype=np.uint8)
        return images, labels

    def __selectModelAndRun(self):
        """
        Selects the model to train up and runs training
        """

        print('\nModel: ', self.modelName)

        # Set up the model if it is keras
        if self.modelName == 'keras':
            self.__setupKeras()

        if self.limit == True:
            print('Training limit is active.')

        # Set up the model if it knn
        if self.modelName == 'knn':
            self.__prepareMachineLearningModel(
                KNeighborsClassifier(n_jobs=4), self.limit)

        # Set up the model if it mlpc
        if self.modelName == 'mlpc':
            self.__prepareMachineLearningModel(
                MLPClassifier(alpha=1, batch_size=128, verbose=1), self.limit)

        # Set up the model if it gaussian
        if self.modelName == 'gaussian':
            self.__prepareMachineLearningModel(
                GaussianNB(), self.limit)

        # Set up the model if it svc
        if self.modelName == 'svc':
            self.__prepareMachineLearningModel(
                SVC(gamma='auto', verbose=1), self.limit)

    def __setupKeras(self):
        # Create model
        self.model = kr.models.Sequential()
        # Add input layer
        self.model.add(Dense(784, activation='relu', input_dim=784))
        # Add output layer
        self.model.add(Dense(10, activation='softmax'))
        # Compile the model
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        # Convert the trainig labels to a binay matrix
        train_labels = kr.utils.to_categorical(self.train_labels, 10)

        # Train the model
        self.model.fit(self.train_images, train_labels,
                       epochs=20, batch_size=128)
        if self.loadTestData and self.checkAccuracy:
            # Convert the test labels to a binay matrix
            test_labels = kr.utils.to_categorical(self.test_labels, 10)
            # Evaluate the code by testing
            print(self.model.evaluate(
                self.test_images, test_labels))

    def __prepareMachineLearningModel(self, model, limited: bool):
        """
        Trains a machine learnong model and calculates accuracy score.
        Prediction is done with 100 test items if limited is on.
        When limited is off then prediction will be done with 10000 items. It can take long time to finish 
        """
        # Check if the model has precit method
        predict = getattr(model, "predict", None)
        if callable(predict):
            # Set the current model to previous model
            self.model = model
            # Train the model
            print("\nTraining model :", model)
            model.fit(self.train_images[0: 500] if limited == True else self.train_images,
                      self.train_labels[0: 500] if limited == True else self.train_labels)
            print("Model trained")
            if self.loadTestData and self.checkAccuracy:
                print("Predicting")
                # Predict test data
                pred = model.predict(
                    self.test_images[0: 500] if limited == True else self.test_images)
                print("Calculating accuracy score")
                # Calculate accuracy
                acc_knn = accuracy_score(
                    self.test_labels[0: 500] if limited == True else self.test_labels, pred)
                print("Model prediction accuracy: %f" % acc_knn)
        else:
            print("Provided model does not have predict method!")

    def __predictWithPreviousModel(self, toBePredicted):
        """
        Tryes to predict a number from the input.
        If there a model is not set yet. It raises and exception
        """
        # Check if a model was loaded
        if hasattr(self, 'model'):
            predicted = self.model.predict(toBePredicted)
            return predicted
        else:
            raise Exception(
                "Call __prepareMachineLearningModel before predictWithPreviousModel")


# Add description
parser = argparse.ArgumentParser(
    description='Recognize handwritten digits on a image')
# Add command line arguments
parser.add_argument('--model', choices=['keras', 'knn', 'mlpc', 'gaussian', 'svc'], default='keras',
                    help='The model to use. One of: kreas, knn, mlpc, gaussian, svc. Default is keras')
parser.add_argument('--verbose', action='store_const',
                    const=True, default=False, help='If flag exist, extra informations is provided about MNIST files')
parser.add_argument('--checkaccuracy',
                    action='store_const', const=True, default=False, help='If flag exist, the trained model will be checked for accuracy ')
parser.add_argument('--limit', action='store_const',
                    const=True, default=False, help='If flag exist, the model will use only 1000 records to train and test. This does not apply for keras!')
parser.add_argument(
    '--savemodel', help='Save the trained model into a file to speed up the application run for next time.')
parser.add_argument(
    '--loadmodel', help='Load trained model from file. This will disregard the --model attribute')
parser.add_argument(
    '--image', help='Path for an image to recognise the number from. It can take a directory path with images in it. If a direcotry path is supplied the last / has to be omitted')
# Parse the models
args = parser.parse_args()
# Create a new object instance
dr = DigitRecognition(args.verbose, args.model,
                      args.limit, args.checkaccuracy, args.savemodel, args.loadmodel, True if args.loadmodel != None or args.loadmodel != "" else False)

# Recognise a number from image(s) if a path is present

image = args.image
# Check if image exists
if image != None:
    # Check if the path is a directory
    if isdir(image) == True:
        # Get files
        for (dirpath, dirnames, filenames) in walk(image):
            # Loop file names
            for f in filenames:
                # Check if file is a picture
                if imghdr.what(image + '/'+f) != None:
                    # Predict the number on the picture
                    dr.recogniseNumberFromPicture(image + '/'+f)
            break
    else:
        # Check if file is a picture
        if imghdr.what(image + '/'+f) != None:
            # Predict number on the picture
            dr.recogniseNumberFromPicture(image)
