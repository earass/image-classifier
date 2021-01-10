import numpy as np
from sklearn.externals import joblib
from Training import data_loader
from skimage.feature import hog

def load_model(filename):
    model = joblib.load(filename)
    return model

def predict(model,Xts):
    preds = model.predict(Xts[0])
    print(preds)
    return preds

def get_test_data():
    # load test data
    Xts = data_loader("TestData.csv")
    print('loaded test data')
    return Xts

def ExtractFeatures(Xts):
    fvs = []
    for X in Xts:
        image = X.reshape([28, 28])
        fd, _ = hog(image, orientations=8, pixels_per_cell=(4, 4),
                            cells_per_block=(1, 1), visualize=True)
        fvs.append(fd)
    return fvs

def execute():
    Xts = get_test_data()
    model = load_model('myModel.pkl')
    Xts = ExtractFeatures(Xts)
    Yts = predict(model, Xts)
    np.savetxt("myPredictions.csv", Yts)
    return

if __name__ == '__main__':
    execute()

