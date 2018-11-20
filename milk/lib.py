import pickle
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from glob import glob
from ncc.models import Model3D
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.path.join(BASE_DIR, 'milk/static')


def load_data(target_dir):
    files = glob(target_dir+'/*/*.jpg')
    model = Model3D(input_shape=(32, 256, 256, 3), num_classes=10)
    from keras.utils import plot_model
    plot_model(model, to_file=MEDIA_ROOT+'/milk/model.png')

    return files, model


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
