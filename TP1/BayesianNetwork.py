import itertools


class BayesianNetwork:
    def __init__(self, dataset, dependency_graph):
        self.dataset = dataset
        self.dependency_graph = dependency_graph
        self.conditional_probabilities = self.get_conditional_probabilities()

    def get_conditional_probabilities(self):
        conditional_probabilities = {}
        for variable in self.dataset.columns:
            conditional_probabilities[variable] = self.get_conditional_probability(variable)
        return conditional_probabilities

    def get_conditional_probability(self, variable):
        parents = list(self.dependency_graph[variable])
        if len(parents) > 0:
            conditional_probability = self.dataset.groupby(parents)[variable] \
                .apply(lambda g: g.value_counts() / len(g)) \
                .reset_index()
            parents.extend([variable, 'probability'])
            conditional_probability.columns = parents
            for _, row in conditional_probability[conditional_probability['probability'] == 1.000000].iterrows():
                for value in list(set(self.dataset[variable].unique()) - {row[variable]}):
                    new_row = row.to_dict()
                    new_row[variable] = value
                    new_row['probability'] = 0.000000
                    conditional_probability = conditional_probability.append(new_row, ignore_index=True)

        else:
            conditional_probability = self.dataset.groupby(variable) \
                .apply(lambda g: len(g) / len(self.dataset[variable])) \
                .reset_index()
            conditional_probability.columns = [variable, 'probability']
        return conditional_probability

    def get_probability(self, request: str):
        values = {}
        variables = list(map(lambda var: var.strip().split('='), request.split(',')))
        for variable in variables:
            name, value = variable
            values[name] = int(value)

        probability = 0
        missing_variables = list(set(self.dataset.columns) - set(values.keys()))
        for combination in itertools.product(*(list(map(lambda missing_variable:
                                                        self.dataset[missing_variable].unique(), missing_variables)))):
            for i in range(len(missing_variables)):
                values[missing_variables[i]] = combination[i]
            probability += self.get_joint_probability(values)
        return probability

    def get_joint_probability(self, values):
        probability = 1
        for variable in values.keys():
            probability_row = self.conditional_probabilities[variable]
            for parent in self.dependency_graph[variable]:
                probability_row = probability_row[probability_row[parent] == values[parent]]
            probability *= probability_row[probability_row[variable] == values[variable]]['probability'].to_list()[0]
        return probability
