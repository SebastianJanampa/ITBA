import numpy as np
class NaiveBayesClassifier:
    def __init__(self, X, y):
        # Establecer datos
        self.x = X
        self.y = y
        # Nombres
        self.variables = X.keys()
        self.target = y.keys()
    def test(self, tests):
        # Outputs
        names = []
        proba = []
        # Inicio del clasificador
        target_names = y.unique()
        if type(tests[0]) is not list:
            tests = [tests]
        for test in tests:
            probs = []
            for case in target_names:
                index = self.y.loc[self.y==case].index
                prob = (self.y == case).mean()
                for var, val in zip(self.variables, test):
                    prob *= (self.x[var].iloc[index] == val).mean()
                probs.append(prob)
            probs = np.array(probs)
            probs /= probs.sum()
            index = np.argmax(probs)
            names.append(target_names[index])
            proba.append(probs[index])
        return names, proba
    
