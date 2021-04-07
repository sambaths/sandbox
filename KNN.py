import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2)**2)

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


class KNN:
    def __init__(self, k: int =3):
        self.k = k

    def fit(self, X, y):    
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predictions = []
        for x in X:
            # compute euclidean distance
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            
            # get k nearest samples and labels
            k_idx = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_idx]

            # majority vote for prediction
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)

if __name__ == "__main__":
    # for getting dataset and splitting data
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    k = 5
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Classification accuracy:", accuracy(y_test, predictions))