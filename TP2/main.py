import argparse
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from TP2.Data import Data
from TP2.ID3 import Tree


def month_categories(months):
    if months < 21:
        return 'corto'
    elif 21 <= months < 38:
        return 'normal'
    elif 38 <= months < 55:
        return 'mediano'
    else:
        return 'largo'


def age_categories(age):
    if age < 33:
        return 'joven'
    elif 33 <= age < 47:
        return 'adulto'
    elif 47 <= age < 61:
        return 'adulto_mayor'
    else:
        return 'jubilado'


def amount_categories(amount):
    if amount < 4794:
        return 'poco'
    elif 4794 <= amount < 9337:
        return 'bastante'
    elif 9337 <= amount < 13881:
        return 'mucho'
    else:
        return 'demasiado'


def german_credit(dataset_path: str, train_percentage: float, goal: str, id2: bool, knn: bool):
    dataset = pd.read_csv(dataset_path)

    # Categorizacion de variables
    # min 4 max 72]
    dataset['Duration of Credit (month)'] = dataset['Duration of Credit (month)'].apply(lambda value: month_categories(value))
    # min 19 max 75]
    dataset['Age (years)'] = dataset['Age (years)'].apply(lambda value: age_categories(value))
    # min 250 max 18424
    dataset['Credit Amount'] = dataset['Credit Amount'].apply(lambda value: amount_categories(value))

    seed = 101
    train, test = train_test_split(dataset, train_size=train_percentage,
                                   test_size=(1 - train_percentage), random_state=seed)
    train_data = Data(train, goal)
    test_data = Data(test, goal)

    Tree(train_data)


def reviews_sentiment(dataset_path: str):
    dataset = pd.read_excel(dataset_path)
    pass


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


def main():
    argparser = argparse.ArgumentParser(description='Aprendizaje Automático: Algoritmos de Clasificación Supervisada',
                                        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument('-e', '--exercise', type=int, choices=[1, 2], required=True)
    argparser.add_argument('-d', '--dataset_path', required=True)
    argparser.add_argument('-t', '--train_percentage', type=float, required=True)
    argparser.add_argument('-o', '--objetivo', required=True)
    argparser.add_argument('-id3', '--id3', action='store_true', required=False)
    argparser.add_argument('-knn', '--knn', action='store_true', required=False)
    args = vars(argparser.parse_args())

    switcher = {
        1: lambda: german_credit(args['dataset_path'], args['train_percentage'], args['objetivo'], args['id3'], args['knn']),
        2: lambda: reviews_sentiment(args['dataset_path'])
    }
    switcher.get(args['exercise'])()


if __name__ == "__main__":
    main()
