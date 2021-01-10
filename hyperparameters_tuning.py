from sklearn.model_selection import GridSearchCV
from sklearn import svm
from Training import get_training_data
from sklearn.neighbors import KNeighborsClassifier

def get_best_hyperparameters(X,y,classifier_object,param_grid):
    grid = GridSearchCV(classifier_object, param_grid, refit=True, verbose=3, cv=5,scoring='accuracy')

    # fitting the model for grid search
    grid.fit(X, y)

    return grid.best_params_,grid.best_estimator_,

def get_param_for_svm(X,y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}
    best_params, best_estimator = get_best_hyperparameters(X, y, svm.SVC(), param_grid)
    print("SVM best param: ", best_params)
    return

def get_param_for_knn(X,y):
    param_grid = dict(n_neighbors = list(range(1,26)))
    best_params, best_estimator = get_best_hyperparameters(X, y, KNeighborsClassifier(), param_grid)
    print("KNN best param: ", best_params)
    return

if __name__ == '__main__':
    Xtr, Ytr, fvs = get_training_data()
    # get_param_for_svm(X=fvs[:10], y=Ytr[:10])

    get_param_for_knn(X=fvs,y=Ytr)
    pass

