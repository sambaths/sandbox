import numpy as np
from collections import Counter
from DecisionTree import DecisionTree

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

class RandomForest:
    def __init__(self, estimators=100, min_samples_split=2, max_depth=100, n_features=None):
        self.estimators=estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.estimators):
            tree = DecisionTree(min_samples_split = self.min_samples_split,
                                max_depth = self.max_depth,
                                n_features= self.n_features)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_preds = [self._most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_preds)

if __name__=="__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metrics import accuracy

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = RandomForest(estimators=3)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy(y_test, y_pred))