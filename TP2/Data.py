import math

from pandas import DataFrame


class Data:
    dataset: DataFrame
    goal: str

    def __init__(self, dataset: DataFrame, goal: str):
        self.dataset = dataset
        self.goal = goal
        self.categories_probabilities = (dataset[goal].value_counts() + 1) / \
                                        (len(dataset[goal]) + len(dataset[goal].unique()))  # Laplace
        self.attributes_probabilities = {}

        self.attributes = list(set(dataset.columns) - {goal})
        for attribute in self.attributes:
            self.attributes_probabilities[attribute] = (dataset[attribute].value_counts() + 1) / \
                                                        (len(dataset[attribute]) + len(dataset[attribute].unique()))

    def gain(self, attribute: str):
        gain = self.entropy()
        for attribute_value in self.attributes_probabilities[attribute].keys():
            subset = Data(self.dataset[self.dataset[attribute] == attribute_value], self.goal)
            gain -= (len(subset.dataset)/len(self.dataset)) * subset.entropy()
        return gain

    def entropy(self):
        entropy = 0
        for probability in self.categories_probabilities.values:
            entropy -= probability * math.log2(probability)
        return entropy

    def subset(self, length: int):
        return self.dataset.sample(n=length, replace=True, axis='index')
