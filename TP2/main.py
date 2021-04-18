import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from TP2.Data import Data
from TP2.ID3 import Tree, KNN


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


def german_credit(dataset_path: str, train_percentage: float, goal: str, id3: bool, random_forest: bool):
    dataset = pd.read_csv(dataset_path)

    # Attributes categorization
    # min 4 max 72
    dataset['Duration of Credit (month)'] = dataset['Duration of Credit (month)']. \
        apply(lambda value: month_categories(value))
    # min 19 max 75
    dataset['Age (years)'] = dataset['Age (years)'].apply(lambda value: age_categories(value))
    # min 250 max 18424
    dataset['Credit Amount'] = dataset['Credit Amount'].apply(lambda value: amount_categories(value))

    seed = 101
    train, test = train_test_split(dataset, train_size=train_percentage,
                                   test_size=(1 - train_percentage), random_state=seed)
    train_data = Data(train, goal)
    test_data = Data(test, goal)

    if id3:
        tree = Tree(train_data, None)
        print_table(tree.confusion_matrix(test_data))
        tree.plot_precision_curve({'Train Dataset': train_data, 'Test Dataset': test_data})

    if random_forest:
        forest = Tree.random_forest(train_data, test_data, 20)
        print_table(forest.confusion_matrix(test_data))
        forest.plot_precision_curve({'Train Dataset': train_data, 'Test Dataset': test_data})


def reviews_sentiment(dataset_path: str):
    df = pd.read_csv(dataset_path, sep=';')
    # Preprocesamiento
    def sentimentNum(text):
        if text == 'negative':
            return 0
        else:
            return 1

    df.textSentiment = df.textSentiment.apply(sentimentNum)
    df.titleSentiment = df.titleSentiment.apply(sentimentNum)
    df.drop(columns=['Review Title', 'Review Text'], inplace=True)
    # Inciso a
    print('Inciso a: %.3f'%df.wordcount[df['Star Rating']==1].mean())
    # Inciso b
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(df.drop(columns=['Star Rating']),
                                                    df['Star Rating'],
                                                    train_size=0.6, random_state=42)
    # Clasificador
    clf = KNN(k=5)
    clf.fit(Xtrain, Ytrain)
    # Sin Pesos
    y_pred = clf.predict(Xtest)
    clf.precision(Ytest.to_numpy(), y_pred)
    clf.conf_matrix(Ytest.to_numpy(), y_pred)
    # Con pesos

def print_table(table):
    rows = list(table.keys())
    columns = list(table[rows[0]].keys())

    matrix = ''.ljust(14)[:14] + ' '
    for col in columns:
        matrix += col.ljust(14)[:14] + ' '

    for row in rows:
        matrix += '\n' + row.ljust(14)[:14] + ' '
        for col in columns:
            matrix += str(table[row][col]).ljust(14)[:14] + ' '

    print(matrix)


def main():
    argparser = argparse.ArgumentParser(description='Aprendizaje Automático: Algoritmos de Clasificación Supervisada',
                                        formatter_class=argparse.RawTextHelpFormatter)

    argparser.add_argument('-e', '--exercise', type=int, choices=[1, 2], required=True)
    argparser.add_argument('-d', '--dataset_path', required=True)
    argparser.add_argument('-t', '--train_percentage', type=float, required=False)
    argparser.add_argument('-o', '--objetivo', required=False)
    argparser.add_argument('-id3', '--id3', action='store_true', required=False)
    argparser.add_argument('-rf', '--random_forest', action='store_true', required=False)
    args = vars(argparser.parse_args())

    switcher = {
        1: lambda: german_credit(args['dataset_path'], args['train_percentage'], args['objetivo'],
                                 args['id3'], args['random_forest']),
        2: lambda: reviews_sentiment(args['dataset_path'])
    }
    switcher.get(args['exercise'])()


if __name__ == "__main__":
    main()
