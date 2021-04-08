import numpy as np
from scipy.special import expit
from metrics import accuracy

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=10000, threshold=0.5):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold=0.5
        self.weights = None
        self.bias = None

    def _sigmoid(self, X):
        # scipy implementation of (1 / (1 + exp(-X)))
        return expit(X)
    
    def fit(self, X, y):
        self._N, self._features = X.shape
        self.weights = np.zeros(self._features)
        self.bias = 0

        for it in range(self.iterations + 1):
            y_hat = self.predict_proba(X)
            self._update_weights(X, y, y_hat)

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        return y_hat

    def predict(self, X):
        y_hat = self.predict_proba(X)
        y_cls = [1 if i > self.threshold else 0 for i in y_hat]
        return y_cls

    def _update_weights(self, X, y, y_hat):
        dw = (2 / self._N) * (np.dot(X.T, (y_hat - y)))
        db = (2 / self._N) * (np.sum(y_hat - y))

        self.weights -= self.learning_rate * dw 
        self.bias -= self.learning_rate * db 

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Classification Accuracy:', accuracy(y_test, preds))