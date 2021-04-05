import numpy as np

class LinearRegression:
    def __init__(self, learning_rate = 0.01, iterations = 10000):
        self.learning_rate = 0.01
        self.iterations = 10000

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
    X = np.random.rand(100, 5)
    real_weights = [1,2,3,4,5]
    real_bias = 20
    y =  np.dot(real_weights, X.T) + real_bias
    regression = LinearRegression()
    w, b = regression.fit(X, y)
    print('Actual: ', real_weights, real_bias)
    print('Trained: ', np.round(w,  2), round(b, 2))

