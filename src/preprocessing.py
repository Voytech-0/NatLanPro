import pandas as pd
import re
import matplotlib.pyplot as plt


def get_csv(filename: str, filepath: str) -> (pd.DataFrame, int):
    with open(filepath) as file:
        text = file.read()

    new_lines = []
    empty_lines = 0
    lines = text.split('\n')
    for line in lines:
        processed_sentence: str = ""
        for word in line.split(' '):
            # get rid of punctuation at the beginning or end of the word
            word = re.sub(r"^\W+|\W+$", "", word)
            # get rid of words '\xa0\xa0'
            if word != '\xa0\xa0' and word != "":
                processed_sentence += word + " "
        if line != "":
            new_lines.append(processed_sentence)
        else:
            empty_lines += 1
    df = pd.DataFrame(new_lines, columns=[filename])

    return df, empty_lines


if __name__ == "__main__":
    print("Hello from preprocessing.py")
    en_df, en_empty_lines = get_csv("en_translation", "../data/europarl-v7.es-en-en.txt")
    es_df, es_empty_lines = get_csv("es_translation", "../data/europarl-v7.es-en-es.txt")

    # barplot with 4 bars
    plt.bar(["english empty lines", "english non-empty lines", "spanish empty lines", "spanish non-empty lines"],
            [en_empty_lines, len(en_df), es_empty_lines, len(es_df)])
    # empty lines are red and non-empty lines are green
    plt.bar(["english empty lines", "spanish empty lines"], [en_empty_lines, es_empty_lines], color="red")
    plt.bar(["english non-empty lines", "spanish non-empty lines"], [len(en_df), len(es_df)], color="blue")
    plt.title("Number of empty and non-empty lines in the english and spanish translations")
    # percentages on top of the bars
    for i, v in enumerate([en_empty_lines, len(en_df), es_empty_lines, len(es_df)]):
        if i < 2:
            plt.text(i, v, f"{round(v/(en_empty_lines + len(en_df))*100, 2)}%")
        else:
            plt.text(i, v, f"{round(v/(es_empty_lines + len(es_df))*100, 2)}%")
    plt.xticks([1, 2], ["English", "Spanish"])
    leg = plt.legend(["Non-empty lines", "Empty lines"], loc="upper right")
    plt.show()

    # make a single dataframe
    df = pd.concat([en_df, es_df], axis=1)
    df.to_csv("../data/en_es_translation.csv", index=False)
