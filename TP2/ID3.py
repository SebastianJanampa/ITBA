from TP2.Data import Data
import matplotlib.pyplot as plt


class Tree:
    node_attribute: str
    node_value: str
    children: dict
    dataset: Data
    height: int

    def __init__(self, dataset: Data, parent_dataset: Data):
        self.height = 1

        # step 1.1 and 1.2
        for value, probability in dataset.categories_probabilities.iteritems():
            if probability == 1.0:
                self.node_value = str(value)
                return

        # step 1.3
        if (len(dataset.dataset)) == 0 or len(dataset.attributes) == 0:
            self.node_value = str(parent_dataset.categories_probabilities.idxmax())
            return

        self.dataset = dataset

        # step 4.1 and 4.2 (get attribute with max gain)
        attribute_with_max_gain = None
        max_gain = 0

        for attribute in dataset.attributes:
            attribute_gain = dataset.gain(attribute)
            if attribute_gain > max_gain:
                max_gain = attribute_gain
                attribute_with_max_gain = attribute

        # step 4.3 (set root)
        self.node_attribute = attribute_with_max_gain
        self.node_value = None
        self.children = {}

        # step 4.4
        for attribute_val in dataset.attributes_probabilities[self.node_attribute].keys():
            children_dataset = Data(dataset.dataset[dataset.dataset[self.node_attribute] == attribute_val]
                                    .drop(columns=[self.node_attribute]), dataset.goal)
            self.children[attribute_val] = Tree(children_dataset, dataset)

        children_max_depth = 0
        for children in self.children.values():
            if children.height > children_max_depth:
                children_max_depth = children.height
        self.height += children_max_depth

    def classify_example(self, example, max_depth: int = None):
        if self.node_value is not None:
            return self.node_value

        if example[self.node_attribute] not in self.children.keys() or max_depth == 0:
            return str(self.dataset.categories_probabilities.idxmax())

        if max_depth is not None:
            max_depth -= 1

        return self.children[example[self.node_attribute]].classify_example(example, max_depth)

    def classify_dataset(self, test_dataset: Data, max_depth: int = None):
        successes = 0
        for _, example in test_dataset.dataset.iterrows():
            result = self.classify_example(example, max_depth)
            if result == str(example[test_dataset.goal]):
                successes += 1
        return float(successes)/len(test_dataset.dataset)

    def confusion_matrix(self, test_dataset: Data):
        confusion_matrix = {str(c): {str(c): 0 for c in self.dataset.categories_probabilities.keys()}
                            for c in self.dataset.categories_probabilities.keys()}

        for _, test_example in test_dataset.dataset.iterrows():
            result = self.classify_example(test_example)
            expected_result = str(test_example[test_dataset.goal])
            confusion_matrix[expected_result][result] += 1

        return confusion_matrix

    def plot_precision_curve(self, datasets: dict, tree_min_depth: int = 0, tree_max_depth: int = None):
        if tree_max_depth is None:
            tree_max_depth = self.height

        colors = ['tab:red', 'tab:blue']
        color_index = 0

        for dataset_name, dataset in datasets.items():
            x, y = [], []

            for depth in range(tree_min_depth, tree_max_depth + 1):
                x.append(depth)
                y.append(self.classify_dataset(dataset, depth))

            plt.plot(x, y, label=dataset_name, color=colors[color_index])
            color_index += 1

        plt.xlabel('Maximum tree depth')
        plt.ylabel('Precision')
        plt.legend(loc='upper left')
        plt.show()
