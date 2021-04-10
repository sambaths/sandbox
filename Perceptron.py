import numpy as np 

class Perceptron:
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation = self._step_function
        self.weights = None
        self.bias = None

    def _step_function(self, x):
        return np.where(x>=0, 1, 0)

    def fit(self, X, y):
        self._N, self._features = X.shape
        self.weights = np.zeros(self._features)
        self.bias = 0
        y_ = np.array([1 if i > 0 else 0 for i in y])
        for it in range(self.iterations):
            for idx, x_i in enumerate(X):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_model)

                update = self.learning_rate * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self.activation(linear_model)
        return y_hat


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split 
    from metrics import accuracy
    
    X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    p = Perceptron()
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)
    print('Classification Accuracy:', accuracy(y_test, predictions))