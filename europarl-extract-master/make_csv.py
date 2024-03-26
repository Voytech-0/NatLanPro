import pandas as pd
import random


def make_csv():
    # go into data folder open file europarl-v7.es-en-en.txt
    with open("corpora/europarl-v7.es-en-en.txt", "r") as f:
        en = f.readlines()
    with open("corpora/europarl-v7.es-en-es.txt", "r") as f:
        es = f.readlines()

    df = pd.DataFrame({"en": en, "es": es})
    df.to_csv("data/europarl.csv", index=False)

    random_index = random.choice(range(len(df)))
    print(df.iloc[random_index]["en"])
    print(df.iloc[random_index]["es"])


if __name__ == "__main__":
    make_csv()
