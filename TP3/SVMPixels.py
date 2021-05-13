import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from PIL import Image


class SVMPixels:
    model: svm.SVC
    classes: list = ['cielo', 'pasto', 'vaca']
    colors: dict = {'cielo': (0, 255, 0), 'pasto': (255, 0, 0), 'vaca': (0, 0, 255)}

    def create_dataset(self, images_path: str):
        images = [images_path + '/cielo.jpg', images_path + '/pasto.jpg', images_path + '/vaca.jpg']
        X = []
        y = []

        for i in range(len(self.classes)):
            image_pixels = list(Image.open(images[i]).getdata())
            X.extend([list(pixel) for pixel in image_pixels])
            y.extend([self.classes[i]] * len(image_pixels))

        return X, y

    def train(self, C, kernel, X, y):
        self.model = svm.SVC(C=C, kernel=kernel)
        self.model.fit(X, y)

    def print_results(self, X, y_real):
        y_predicted = self.model.predict(X)

        confusion_matrix = {c: {c: 0 for c in self.classes} for c in self.classes}
        examples = len(y_predicted)
        corrects = 0

        for i in range(examples):
            confusion_matrix[y_real[i]][y_predicted[i]] += 1
            if y_real[i] == y_predicted[i]:
                corrects += 1

        print_table(confusion_matrix)
        print('\nModels precision: %f\n\n' % (corrects/examples))

    def classify_image(self, images_path: str):
        image = images_path + '/cow.jpg'
        image_pixels = list(Image.open(image).getdata())
        results = self.model.predict(image_pixels)

        image = np.array(Image.open(image))
        plt.imshow(image)

        index = 0
        for x, y in np.ndindex(image.shape[:-1]):
            image[x][y] = self.colors[results[index]]
            index += 1

        plt.imshow(image, alpha=0.4)
        plt.show()


def print_table(table):
    rows = list(table.keys())
    columns = list(table[rows[0]].keys())

    matrix = ''.ljust(10)[:10] + ' '
    for col in columns:
        matrix += col.ljust(10)[:10] + ' '

    for row in rows:
        matrix += '\n' + row.ljust(10)[:10] + ' '
        for col in columns:
            matrix += str(table[row][col]).ljust(10)[:10] + ' '

    print(matrix)
