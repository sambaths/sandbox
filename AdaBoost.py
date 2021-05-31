import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        self._N = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(self._N)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        
        return predictions

class AdaBoost:
    def __init__(self, estimators=5):
        self.estimators = estimators
    
    def fit(self, X, y):
        self._N, self._features = X.shape
        
        w = np.full(self._N, (1/self._N))

        self.estimator_list = []
        for _ in range(self.estimators):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(self._features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(self._N)
                    predictions[X_column < threshold] = -1

                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
            
            epsilon = 1e-10
            clf.alpha = 0.5 * np.log((1-min_error) / (min_error + epsilon))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            self.estimator_list.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.estimator_list]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metrics import accuracy

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    y[y==0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    
    clf = AdaBoost()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print('Accuracy: ', accuracy(y_test, y_pred))