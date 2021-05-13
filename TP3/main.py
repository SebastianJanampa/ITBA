import argparse
from sklearn.model_selection import train_test_split

from TP3.SVMPixels import SVMPixels


def classify_points():
    pass


def classify_pixels(images_path: str, train_size: float):
    SVM = SVMPixels()
    X, y = SVM.create_dataset(images_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=(1 - train_size), random_state=101)

    # Finding best parameters
    # kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # Cs = [0.01, 0.1, 1.0, 10.0]
    # for kernel in kernels:
    #     for c in Cs:
    #         print('SVM with %s kernel and C = %f:\n' % (kernel, c))
    #         SVM.train(c, kernel, X_train, y_train)
    #         SVM.print_results(X_test, y_test)

    best_kernel = 'rbf'
    best_c = 10.0
    SVM.train(best_c, best_kernel, X_train, y_train)
    SVM.classify_image(images_path)


def main():
    argparser = argparse.ArgumentParser(description='Aprendizaje Automático: Algoritmos de Vectores de Soporte',
                                        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument('-e', '--exercise', type=int, choices=[1, 2], required=True)
    argparser.add_argument('-d', '--images_path', required=False)
    argparser.add_argument('-t', '--train_size', type=float, required=False)
    args = vars(argparser.parse_args())

    switcher = {
        1: lambda: classify_points(),
        2: lambda: classify_pixels(args['images_path'], args['train_size'])
    }
    switcher.get(args['exercise'])()


if __name__ == "__main__":
    main()
