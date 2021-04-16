from TP2.Data import Data


class Tree:
    node_value: str
    dataset: Data

    def __init__(self, dataset: Data):
        # step 1.1 and 1.2
        for value, probability in dataset.categories_probabilities.iteritems():
            if probability == 1.0:
                self.node_value = str(value)
                return

        # step 1.3
        if (len(dataset.dataset)) == 0:
            self.node_value = str(dataset.categories_probabilities.idxmax)

        self.dataset = dataset

        # step 4.1

