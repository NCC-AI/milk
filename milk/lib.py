import pickle
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from glob import glob
import os

from ncc.models import Model3D, Model2D
from ncc.preprocessing import preprocess_input, get_dataset
from ncc.validations import save_show_results, evaluate
from ncc.readers import search_from_annotation
from ncc.generators import generate_arrays_from_annotation

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

from keras.callbacks import EarlyStopping
from keras.utils import plot_model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.path.join(BASE_DIR, 'milk/static')


def load_data(target_dir):
    test_size = 0.2

    if os.path.isdir(target_dir+'/train') and os.path.isdir(target_dir+'/test'):
        x_train, y_train = get_dataset(target_dir+'/train')
        x_test, y_test = get_dataset(target_dir+'/test')

    else:
        x_train, y_train = get_dataset(target_dir)

    if x_test is None and y_test is None:
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)

    # prepare data
    x_train, y_train = preprocess_input(x_train, y_train)
    x_test, y_test = preprocess_input(x_test, y_test)
    print(x_train.shape, y_train.shape)

    # data profile
    class_names = list(np.arange(y_train.shape[1]))

    num_classes = len(class_names)
    input_shape = x_train.shape[1:]

    # build model
    if len(input_shape) == 3:  # (height, width, channel)
        model = Model2D(input_shape=input_shape, num_classes=num_classes)
    elif len(input_shape) == 4:  # (depth, height, width, channel)
        model = Model3D(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError('input shape is invalid.')

    plot_model(model, to_file=MEDIA_ROOT+'/milk/model.png')
    
    # training parameter
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['acc']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return (x_train, y_train), (x_test, y_test), model


def read():
    with open('mnist.pickle', 'rb') as file:
        clf = pickle.load(file)
    return clf


def create_and_save():
    mnist = datasets.fetch_mldata('MNIST original', data_home='image/')
    X = mnist.data / 255
    y = mnist.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, test_size=0
    )

    clf = svm.SVC()
    clf.fit(X_train, y_train)

    with open('mnist.pickle', 'wb') as file:
        pickle.dump(clf, file)
    return clf


try:
    clf = read()
except FileNotFoundError:
    clf = create_and_save()


def predict(img_array):
    result = clf.predict(img_array)
    return str(int(result[0]))
