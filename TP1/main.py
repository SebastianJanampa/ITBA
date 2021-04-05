import argparse
import pandas as pd
import openpyxl
from NaiveBayesClassifier import NaiveBayesClassifier
from BayesianNetwork import BayesianNetwork
from sklearn.model_selection import train_test_split


def british_preferences(dataset_path: str, scones: bool, cerveza: bool, whisky: bool, avena: bool, futbol: bool):
    ## Establecer variables
    dataset = pd.read_excel(dataset_path)
    x = [scones, cerveza, whisky, avena, futbol]
    variables = dataset.keys().drop('Nacionalidad')
    results = {}
    ## Algoritmo Naive Bayes
    for case in dataset['Nacionalidad']:
        index = dataset.loc[dataset['Nacionalidad']==case].index
        den = len(index) 
        prob = (dataset['Nacionalidad'] == case).sum()
        for var, val in zip(variables, x):
            prob *= (dataset[var].iloc[index] == val).sum()/den
        results[prob] = case
    best = max(results.keys())
    print('Dado los datos %s, hay una mayor probabilidad de que el sujeto sea %s\n\n'%(x, results[best]))



def argentine_news(dataset_path: str):
    dataset = pd.read_excel(dataset_path)
    X = dataset.iloc[:, :3]
    y = dataset.iloc[:, 3]
    train_percentage = 0.9
    seed = 101
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage,test_size=1-train_percentage, random_state=seed)

    classifier = NaiveBayesClassifier(X_train, y_train)
    classifier.train_ej_2()

    # successes, errors, raw_results, expected_results = classifier.test(X_test)
    raw_results = classifier.test_ej_2(X_test)
    print(raw_results)
    # conf_matrix = confusion_matrix(classifier.classes)
    # printTable(conf_matrix)
    # printTable(calculateMetrics(conf_matrix))
    # drawRocCurve()


def admissions(dataset_path: str):
    dataset = pd.read_csv(dataset_path)
    dataset['gre'] = dataset['gre'].apply(lambda value: 1 if value >= 500 else 0)
    dataset['gpa'] = dataset['gpa'].apply(lambda value: 1 if value >= 3 else 0)

    dependency_graph = {'admit': ['gre', 'rank', 'gpa'], 'gre': ['rank'], 'gpa': ['rank'], 'rank': []}

    BayesianNetwork(dataset, dependency_graph)


def confusion_matrix(classes):
    confusion_matrix = {c: {c: 0 for c in classes} for c in classes}
    # for print: matrix += str(table[row][col]).ljust(14)[:14] + ' '


def calculateMetrics(confusion_matrix):
    metricsTable = {}
    for variable in confusion_matrix.keys():
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        Total = 0
        for row in confusion_matrix:
            for col in confusion_matrix[row]:
                Total += confusion_matrix[row][col]
                if row == variable and row == col:
                    TP += confusion_matrix[row][col]
                elif col == variable:
                    FP += confusion_matrix[row][col]
                elif row == variable:
                    FN += confusion_matrix[row][col]
                else:
                    TN += confusion_matrix[row][col]
        accuracy = (TP + TN) / (TP + TN + FN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        TPrate = TP / (TP + FN)
        FPrate = FP / (FP + TN)
        F1score = (2 * precision * recall) / (precision + recall)
        metricsTable[variable] = {'Accuracy': accuracy, 'Precision': precision, 'TP Rate': TPrate, 'FP Rate': FPrate, 'F1-Score': F1score}
        
    return metricsTable


def printTable(table):
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
    args = vars(argparser.parse_args())

    switcher = {
        1: lambda: british_preferences(args['dataset_path'], args['scones'], args['cerveza'], args['whisky'], args['avena'], args['futbol']),
        2: lambda: argentine_news(args['dataset_path']),
        3: lambda: admissions(args['dataset_path']),
    }
    switcher.get(args['exercise'])()


if __name__ == "__main__":
    main()
