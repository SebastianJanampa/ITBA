import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class KNN:
    k: int
    Xtrain: np.ndarray
    Ytrain: np.ndarray

    def __init__(self, k):
        self.k = k

    def fit(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain.to_numpy()
        self.Ytrain = Ytrain.to_numpy()

    def precision(self, y_true, y_pred):
        cases = [1, 2, 3, 4, 5]
        for case in cases:
            TP = (np.where(y_pred == case, case, 0) == y_true).sum()
            FP = (y_pred == case).sum() - TP
            FN = (y_true == case).sum() - TP
            TN = len(y_pred) - (TP + FP + FN)
            pr = TP / (TP + FP)
            ac = (TP + TN) / len(y_pred)
            re = TP / (TP + FN)
            f1 = 2 * TP / (2 * TP + FP + FN)
            print('FOR %i: precision %.2f, accuracy %.2f, recall %.2f, F1 %.2f'
                  % (case, pr, ac, re, f1))

    def conf_matrix(self, y_true, y_pred):
        classes = [1, 2, 3, 4, 5]
        matrix1 = {c: {c: 0 for c in classes} for c in classes}
        matrix2 = np.zeros([len(classes), len(classes)])
        for i, j in zip(y_true, y_pred):
            matrix1[j][i] += 1
            matrix2[int(j - 1), int(i - 1)] += 1
        sns.heatmap(matrix2, annot=True)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.show()
        return matrix1

    def predict(self, X, weights=False):
        distances = self.calculate_distances(X.to_numpy())
        return self.predict_labels(distances, weights)

    def calculate_distances(self, X):
        """
        return:  (Xtest-Xtrain)**2
        """
        num_test = len(X)
        num_train = len(self.Xtrain)
        distances = np.zeros((num_test, num_train))
        err = 1e-16
        for i in range(num_test):
            distances[i, :] = ((X[i] - self.Xtrain) ** 2).sum(axis=1) + err
        return distances

    def predict_labels(self, distances, weights=False):
        indices = np.argsort(distances, axis=1)
        k_closest_classes = np.array(self.Ytrain[indices[:, :self.k]], dtype=np.int64)
        if weights:
            num_test = distances.shape[0]
            count_classes = np.zeros((num_test, 6))
            k_closest_distances = 1 / np.sort(distances)[:, :self.k]
            for case in range(1, 6):
                count_classes[:, case] = (k_closest_distances * (k_closest_classes == case)).sum(axis=1)
        else:
            count_classes = np.apply_along_axis(lambda x: np.bincount(x, minlength=6),
                                                axis=1, arr=k_closest_classes)
        y_pred = np.argmax(count_classes, axis=1)
        return y_pred
