import pandas as pd
import matplotlib.pyplot as plt


def eliminate_long_sentences(data: pd.DataFrame, max_length: int = 400) -> pd.DataFrame:
    """
    Eliminate long sentences from a dataframe that contains two columns: 'en' and 'es'.
    If any of the sentences in the 'en' or 'es' columns has more than max_length words,
    the row is removed from the dataframe.
    :param data: a dataframe with two columns: 'en' and 'es'
    :param max_length: the maximum length of a sentence
    """
    mask = (data['en'].str.split().str.len() <= max_length) & (data['es'].str.split().str.len() <= max_length)
    return data.loc[mask]


if __name__ == "__main__":
    print("Hello from feat_engineering.py")
    old_df = pd.read_csv("../europarl-extract-master/corpora/preprocessed_europarl.csv")
    df = eliminate_long_sentences(old_df)
    # make a boxplot of the lengths of the sentences
    en_lengths = [len(str(sentence).split(' ')) for sentence in df["en"]]
    es_lengths = [len(str(sentence).split(' ')) for sentence in df["es"]]
    plt.boxplot([en_lengths, es_lengths])
    plt.title("Length of the sentences after removing long sentences")
    plt.xticks([1, 2], ["English", "Spanish"])
    plt.ylabel("Number of words")
    plt.show()

    # save the dataframe to a csv file
    df.to_csv("../europarl-extract-master/corpora/cropped_en_es_translation.csv", index=False)
