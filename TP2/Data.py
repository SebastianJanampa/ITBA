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

        attributes = list(set(dataset.columns) - {goal})
        for attribute in attributes:
            self.attributes_probabilities[attribute] = (dataset[attribute].value_counts() + 1) / \
                                                        (len(dataset[attribute]) + len(dataset[attribute].unique()))
