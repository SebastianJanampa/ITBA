import numpy as np
from operator import itemgetter
class NaiveBayesClassifier:
    def __init__(self, X, y):
        # Data
        self.X = X
        self.y = y

        # Names
        self.variables = self.X.keys()
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
        self.separate_sources(self.X)
        self.X = self.X.drop(columns=['titular', 'fuente', 'fecha'], axis=1)
        self.variables.remove('titular')
        self.variables.remove('fuente')
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


    def count_words(self, df):
        # Count total number of words
        return np.array([len(words) for words in df]).sum()


    def separate_sources(self, df):
        for source in df['fuente'].unique():
            df['source_'+str(source)] = ((df['fuente'] == source) * 1)


    def train_ej_2(self):
        self.preprocess_data()
        self.train()
    

    def train(self):
        ## Establecer variables
        # x = [scones, cerveza, whisky, avena, futbol]
        variables = self.X.keys()
        target = self.y

        ## Algoritmo Naive Bayes
        for case in target.unique():
            cases = self.X.loc[target==case]

            # Generate frequencies
            for var in variables:
                # Calculate probability
                if 'word_' in var:
                    length = self.count_words(cases) + 1  # LaPlace correction
                else:
                    length = len(cases) + 1  # LaPlace correction
                prob = cases[var].sum() / length

                # Add to probabilities
                if case not in self.probabilities:
                    self.probabilities[case] = {}
                self.probabilities[case][var] = prob


    def preprocess_tests(self, tests):
        # Parse `titular` column into word columns, each with the amount of times the word appears
        words_columns = self.X.loc[:, self.X.columns.str.startswith('word_')].keys()
        for word in words_columns:
            aux = word[5:]  # take out 'word_' from word
            self.X[word] = (tests['titular'].str.count(aux))  # amount of times `word` appears in 'titular' column of the same row

        # Parse `fuente` column into a column for each source, with a 1 if was that source and 0 otherwise
        self.separate_sources(tests)

        tests = tests.drop(columns=['titular', 'fuente', 'fecha'], axis=1)

        return tests


    def test_ej_2(self, tests):
        # TODO: test, do ROC and other metrics
        tests = self.preprocess_tests(tests)

        # Output
        tests_prob = []

        # Inicio del clasificador
        # TODO: contar todas las veces que aparece en el titulo
        target_names = self.y.unique()
        # if type(tests[0]) is not list:
        #     tests = [tests]
        for test in tests[:1]:
            probs = []
            for case in target_names:
                prob = (tests == case).mean()  # TODO: check
                for var in self.variables:
                    if var not in self.probabilities[case]:
                        continue
                    prob *= self.probabilities[case][var]
                probs.append(prob)
            probs = np.array(probs)
            # probs /= probs.sum()
            index = np.argmax(probs)
            tests_prob.append({'names': target_names[index], 'prob': probs[index]})
        
        return tests_prob
    
