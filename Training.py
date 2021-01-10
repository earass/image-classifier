import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt

from skimage.feature import hog

def get_training_data():
    # load training examples
    Xtr = data_loader("TrainData.csv")
    print('loaded training examples')

    # load training labels
    Ytr = data_loader("TrainLabels.csv")
    print('loaded training labels')

    # get feature vectors
    fvs = ExtractFeatures(Xtr)
    print("extracted features")

    return Xtr, Ytr, fvs

def data_loader(filename):
    data = np.loadtxt(filename)
    return data

def ExtractFeatures(Xtr):
    fvs = []
    for X in Xtr:
        image = X.reshape([28, 28])
        fd, _ = hog(image, orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True)
        fvs.append(fd)
    return fvs

def train(X,y):
    k = 13
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    return knn

def save_model(model,filename):
    joblib.dump(model, filename)
    return

def execute():
    Xtr, Ytr, fvs = get_training_data()
    model = train(fvs, Ytr)
    save_model(model, 'myModel.pkl')
    print('model saved')

if __name__ == '__main__':
    execute()
