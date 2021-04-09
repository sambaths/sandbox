import numpy as np

class NaiveBayes:
    def __init__(self):
        pass
    def fit(self, X, y):
        self._N, self._features = X.shape
        self._classes = np.unique(y)
        self.n_classes = len(self._classes)

        self._mean = np.zeros((self.n_classes, self._features), dtype=np.float)
        self._var = np.zeros((self.n_classes, self._features), dtype=np.float)
        self._priors = np.zeros(self.n_classes, dtype=np.float)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(self._N)
        

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self._classes):
                prior = np.log(self._priors[idx])
                class_conditional = np.sum(np.log(self._pdf(idx, x)))
                posterior = prior + class_conditional
                posteriors.append(posterior)
            y_pred.append(self._classes[np.argmax(posteriors)])
        return y_pred
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator / denominator


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metrics import accuracy
    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)
    print('Classification Accuracy: ', accuracy(y_test, predictions))