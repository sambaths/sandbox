import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, iterations=10000):
        self.learning_rate = 0.01
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self._N, self._features = X.shape

        self.weights = np.zeros(self._features)
        self.bias = 0

        for it in range(self.iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * (y_[idx])

    def predict(self, X):
        linear_model = np.dot(X, self.weights) - self.bias
        return np.sign(linear_model)

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metrics import accuracy

    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y==0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = SVM()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Classification Accuracy:', accuracy(y_test, preds))