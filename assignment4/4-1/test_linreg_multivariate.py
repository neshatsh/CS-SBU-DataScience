import numpy as np
from keras import backend
from linreg import LinearRegression


def mean_squared_error(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


if __name__ == "__main__":
    '''
        Main function to test multivariate linear regression
    '''

    # load the data
    filePath = 'data/multivariateData.dat'
    file = open(filePath, 'r')
    allData = np.loadtxt(file, delimiter=',')

    X = np.matrix(allData[:, :-1])
    y = np.matrix((allData[:, -1])).T

    n, d = X.shape

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Add a row of ones for the bias term
    X = np.c_[np.ones((n, 1)), X]

    # initialize the model
    init_theta = np.matrix(np.random.randn((d + 1))).T
    n_iter = 2000
    alpha = 0.01

    # Instantiate objects
    lr_model = LinearRegression(init_theta=init_theta, alpha=alpha, n_iter=n_iter)
    lr_model.fit(X, y)

    test_path = r'data/holdout.npz'

    test = np.load(test_path)['arr_0']
    X_test = np.matrix(test[:, :-1])
    y_test = np.matrix(test[:, -1]).T

    X_test = (X_test - mean) / std

    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    y_test_predict = lr_model.predict(X_test)
    error = mean_squared_error(y_test, y_test_predict)
    print(error)
