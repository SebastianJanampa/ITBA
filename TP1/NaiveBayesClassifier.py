import numpy as np
from operator import itemgetter
class NaiveBayesClassifier:
    def __init__(self, X, y):
        # Data
        self.X = X
        self.X_copy = X.copy(deep=True)
        self.y = y

        # Names
        self.variables = list(self.X.keys())
        self.target = self.y.keys()

        # Train data
        self.probabilities = {}  # {'Nacional': {'Clarin.com': 0.1, ...}, 'Deportes': {'Clarin.com': 0.03, ...}, ...}
        self.n_keywords = 5


    def preprocess_data(self):
        """
            Transforms 'Noticias_argentinas' into a binary table that is able
            to be trained by the NaiveBayesClassifier
        """
        self.obtain_key_words_ej_2()
        self.X = self.X.drop(columns=['titular', 'fecha'], axis=1)
        self.variables.remove('titular')
        self.variables.remove('fecha')
    
    def obtain_key_words_ej_2(self):
        words_count = {}
        titular = self.X['titular']
        
        blacklist = ['para', 'tras', 'sobre', 'cómo', 'entre', 'contra', 'nuevo', 'como', 'está', 'tres',
                     'tiene', 'desde', 'este', 'hasta', 'todo', 'baja', 'dijo', 'podría', 'puede', 'pero']
        # Count words repetition
        for row in titular:
            words = [word.lower() for word in row.split(' ')]
            for word in words:
                if len(word) < 4 or word in blacklist: continue
                if word not in words_count:
                    words_count[word] = 1
                else:
                    words_count[word] += 1
        
        # Get `n` most repeated
        most_repeated = dict(sorted(words_count.items(), key = itemgetter(1), reverse = True)[:self.n_keywords]).keys()
        for word in most_repeated:
            self.X['word_'+str(word)] = (self.X['titular'].str.count(word))  # amount of times `word` appears in 'titular' column of the same row
            self.variables.append('word_'+str(word))


    def count_words(self, df):
        # Count total number of words
        return np.array([len(words) for words in df.titular]).sum()


    def train_ej_2(self):
        self.preprocess_data()
        self.train()
    

    def train(self):
        target = self.y

        ## Algoritmo Naive Bayes
        for case in target.unique():
            if case not in self.probabilities:
                self.probabilities[case] = {}
            cases = self.X.loc[target==case]

            # Generate frequencies
            for var in self.variables:
                # Calculate probability
                if 'word_' in var:
                    length = self.count_words(self.X_copy.loc[target==case])  # LaPlace correction
                    prob = cases[var].count() / length
                    self.probabilities[case][var] = prob
                elif var == 'fuente':
                    length = len(cases)  # LaPlace correction
                    for source in cases[var].unique():
                        amount = cases.loc[cases['fuente']==source].count()['fuente']
                        prob = amount / length
                        self.probabilities[case][var] = prob



    def preprocess_tests(self, tests):
        # Parse `titular` column into word columns, each with the amount of times the word appears
        words_columns = self.X.loc[:, self.X.columns.str.startswith('word_')].keys()
        for word in words_columns:
            aux = word[5:]  # take out 'word_' from word
            tests[word] = (tests['titular'].str.count(aux))  # amount of times `word` appears in 'titular' column of the same row

        tests = tests.drop(columns=['titular', 'fecha'], axis=1)
        return tests


    def test_ej_2(self, tests):
        tests = self.preprocess_tests(tests)

        # Output
        tests_prob = []

        # Inicio del clasificador
        # TODO: contar todas las veces que aparece en el titulo
        target_names = self.y.unique()
        # if type(tests[0]) is not list:
        #     tests = [tests]
        for test in tests.iterrows():
            probs = {}
            for case in target_names:
                prob = (self.y == case).mean()  # TODO: check
                for var in test.variables:
                    if var in self.probabilities[case]:
                        prob *= self.probabilities[case][var]
                probs[case] = prob
            # probs = np.array(probs)
            # probs /= probs.sum()
            # index = np.argmax(probs)
            # tests_prob.append({'names': target_names[index], 'prob': probs[index]})
            tests_prob.append(probs)

        return tests_prob
    
