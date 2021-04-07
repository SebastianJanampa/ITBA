import argparse
import pandas as pd
import openpyxl
from NaiveBayesClassifier import NaiveBayesClassifier
from BayesianNetwork import BayesianNetwork
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def british_preferences(dataset_path: str, scones: bool, cerveza: bool, whisky: bool, avena: bool, futbol: bool):
    # Establecer variables
    dataset = pd.read_excel(dataset_path)
    x = [scones, cerveza, whisky, avena, futbol]
    variables = dataset.keys().drop('Nacionalidad')
    results = {}
    # Algoritmo Naive Bayes
    for case in dataset['Nacionalidad']:
        index = dataset.loc[dataset['Nacionalidad'] == case].index
        prob = (dataset['Nacionalidad'] == case).mean()
        for var, val in zip(variables, x):
            prob *= (dataset[var].iloc[index] == val).mean()
        results[prob] = case
    best = max(results.keys())
    print('Dado los datos %s, hay una mayor probabilidad de que el sujeto sea %s\n\n' % (x, results[best]))


def argentine_news(dataset_path: str):
    dataset = pd.read_excel(dataset_path)
    dataset.dropna(inplace=True)
    dataset = dataset[
        (dataset['categoria'] == 'Salud') |
        (dataset['categoria'] == 'Deportes') |
        (dataset['categoria'] == 'Economia') |
        (dataset['categoria'] == 'Entretenimiento') |
        (dataset['categoria'] == 'Internacional')
    ]
    dataset.drop_duplicates(inplace=True)
    X = dataset.iloc[:, :3]
    y = dataset.iloc[:, 3]
    train_percentage = 0.9
    seed = 101
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage,
                                                        test_size=1 - train_percentage, random_state=seed)

    classifier = NaiveBayesClassifier(X_train, y_train)
    classifier.train_ej_2()

    # successes, errors, raw_results, expected_results = classifier.test(X_test)
    raw_results = classifier.test_ej_2(X_test)  # raw_results: [{'categoria1': prob, 'categoria2': prob, ..., 'categoriaN': prob}, ...] one dict for each row

    expected_results = y_test.dropna()
    expected_results = expected_results.reset_index(drop=True)
    classes = [a for a in y_train.unique()]
    conf_matrix = confusion_matrix(classes, raw_results, expected_results)
    print_table(conf_matrix)
    print()
    print_table(calculateMetrics(conf_matrix))
    drawRocCurve(raw_results, expected_results, 'Salud')


def admissions(dataset_path: str, probability_request: str):
    dataset = pd.read_csv(dataset_path)
    dataset['gre'] = dataset['gre'].apply(lambda value: 1 if value >= 500 else 0)
    dataset['gpa'] = dataset['gpa'].apply(lambda value: 1 if value >= 3 else 0)

    dependency_graph = {'admit': ['gre', 'rank', 'gpa'], 'gre': ['rank'], 'gpa': ['rank'], 'rank': []}

    bn = BayesianNetwork(dataset, dependency_graph)
    request_elements = probability_request.split('|')
    if len(request_elements) == 1:
        probability = bn.get_probability(request_elements[0])
    else:
        probability = bn.get_probability(request_elements[0] + ',' + request_elements[1]) \
                      / bn.get_probability(request_elements[1])
    print(f'P({probability_request}) = {probability}')


def confusion_matrix(categories, raw_results, expected_results):
    print()
    categories = [c for c in categories if isinstance(c, str)]
    confusion_matrix = {c: {c: 0 for c in categories} for c in categories}
    for j, _ in expected_results.iteritems():
        predictions = raw_results[j]
        expected_result = expected_results[j]
        predicted_result = max(predictions, key=lambda i: predictions[i])
        confusion_matrix[expected_result][predicted_result] += 1
    return confusion_matrix


def drawRocCurve(raw_results, expected_results, class_name):
    x = []
    y = []
    thoughputs = []
    for i in range(0, 11):
        throughput = i / 10
        TP = 0
        FN = 0
        FP = 0
        TN = 0
        for j in range(len(raw_results)):
            probability = raw_results[j][class_name]
            expected_result = expected_results[j]
            if probability >= throughput:
                if expected_result == class_name:
                    TP += 1
                else:
                    FP += 1
            else:
                if expected_result == class_name:
                    FN += 1
                else:
                    TN += 1
        FPrate = FP / (FP + TN)
        TPrate = TP / (TP + FN)
        x.append(FPrate)
        y.append(TPrate)
        thoughputs.append(throughput)
    plt.plot(x, y, '-or')
    for j, throughput in enumerate(thoughputs):
        plt.annotate(str(throughput), (x[j], y[j]))
    plt.show()


def calculateMetrics(confusion_matrix):
    metricsTable = {}
    for attr in confusion_matrix.keys():
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        total = 0
        for row in confusion_matrix:
            for col in confusion_matrix[row]:
                value = confusion_matrix[row][col]
                total += value
                if row == attr and col == attr:
                    TP += value
                elif col == attr:
                    FP += value
                elif row == attr:
                    FN += value
                else:
                    TN += value
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1score = (2 * precision * recall) / (precision + recall)
        TPrate = TP / (TP + FN)
        FPrate = FP / (FP + TN)
        metricsTable[attr] = {'Accuracy': accuracy, 'Precision': precision, 'Tasa TP': TPrate, 'Tasa FP': FPrate, 'F1-score': F1score}
        
    return metricsTable


def print_table(table):
    rows = list(table.keys())
    columns = list(table[rows[0]].keys())
    row_len = 15

    matrix = ''.ljust(row_len)[:row_len] + ' '
    for col in columns:
        matrix += col.ljust(row_len)[:row_len] + ' '

    for row in rows:
        matrix += '\n' + row.ljust(row_len)[:row_len] + ' '
        for col in columns:
            matrix += str(table[row][col]).ljust(row_len)[:row_len] + ' '

    print(matrix)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "True", "Yes")


def main():
    argparser = argparse.ArgumentParser(description='Aprendizaje Automático: Método de Bayes',
                                        formatter_class=argparse.RawTextHelpFormatter)
    argparser.register('type', 'bool', str2bool)
    argparser.add_argument('-e', '--exercise', type=int, choices=[1, 2, 3], required=True)
    argparser.add_argument('-d', '--dataset_path', required=True)
    argparser.add_argument('-s', '--scones', type='bool', choices=[True, False], required=False)
    argparser.add_argument('-c', '--cerveza', type='bool', choices=[True, False], required=False)
    argparser.add_argument('-w', '--whisky', type='bool', choices=[True, False], required=False)
    argparser.add_argument('-a', '--avena', type='bool', choices=[True, False], required=False)
    argparser.add_argument('-f', '--futbol', type='bool', choices=[True, False], required=False)
    argparser.add_argument('-r', '--probability_request', required=False)
    args = vars(argparser.parse_args())

    switcher = {
        1: lambda: british_preferences(args['dataset_path'], args['scones'], args['cerveza'], args['whisky'],
                                       args['avena'], args['futbol']),
        2: lambda: argentine_news(args['dataset_path']),
        3: lambda: admissions(args['dataset_path'], args['probability_request']),
    }
    switcher.get(args['exercise'])()


if __name__ == "__main__":
    main()
