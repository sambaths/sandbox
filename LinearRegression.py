import numpy as np
from metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 10000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def predict(self, X):
        return X.dot(self.w) + self.b

    def update_weights(self, X, y, y_hat):
        db = - 2 * np.sum(y - y_hat) / self._N
        dW = - (2 * (X.T).dot(y - y_hat)) / self._N
        self.w = self.w - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self.w, self.b
    
    def loss(self, y_hat, y):
        C = 1 / self._N * np.sum(np.power(y - y_hat, 2))
        return C


    def fit(self, X, y):
        self._N, self._features = X.shape

        self.w = np.zeros(self._features)
        self.b = 0

        for it in range(self.iterations + 1):
            y_hat = self.predict(X)
            loss = self.loss(y_hat, y)

            if it % 1000 == 0:
                print(f'Loss at iteration {it} is {round(loss, 4)}')
            
            self.w, self.b = self.update_weights(X, y, y_hat)

        return self.w, self.b
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
        
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)