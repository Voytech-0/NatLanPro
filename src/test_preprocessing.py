import random
import pandas as pd


class MyTestCase:

    def setUp(self):
        self.df = pd.read_csv('../data/en_es_translation.csv')
        with open("../europarl-extract-master/corpora/europarl-v7.es-en-en.txt") as file:
            text = file.read()
            lines = text.split('\n')
            self.en_df = pd.DataFrame(lines)
        with open("../europarl-extract-master/corpora/europarl-v7.es-en-es.txt") as file:
            text = file.read()
            lines = text.split('\n')
            self.es_df = pd.DataFrame(lines)


    def test_something(self):
        random_index = random.randint(0, len(self.df) - 1)
        random_row = self.df.iloc[random_index]
        print(random_row[0])
        print(random_row[1])


if __name__ == '__main__':
    test = MyTestCase()
    test.setUp()
    test.test_something()
