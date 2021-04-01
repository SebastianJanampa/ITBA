import argparse
import pandas as pd
import openpyxl

from TP1.BayesianNetwork import BayesianNetwork


def british_preferences(dataset_path: str, scones: bool, cerveza: bool, whisky: bool, avena: bool, futbol: bool):
    dataset = pd.read_excel(dataset_path)
    pass


def argentine_news(dataset_path: str):
    dataset = pd.read_excel(dataset_path)
    pass


def admissions(dataset_path: str):
    dataset = pd.read_csv(dataset_path)
    dataset['gre'] = dataset['gre'].apply(lambda value: 1 if value >= 500 else 0)
    dataset['gpa'] = dataset['gpa'].apply(lambda value: 1 if value >= 3 else 0)

    dependency_graph = {'admit': ['gre', 'rank', 'gpa'], 'gre': ['rank'], 'gpa': ['rank'], 'rank': []}

    BayesianNetwork(dataset, dependency_graph)


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
