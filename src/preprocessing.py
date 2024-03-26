import pandas as pd
import re
import matplotlib.pyplot as plt
import random


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # remove punctuation at the beginning and end of the words
    df = df.map(lambda x: re.sub(r"^\W+|\W+$", "", x))
    return df


def plot_empty_lines(df: pd.DataFrame):
    es_empty_lines = len(df[df.iloc[:, 0] == ""])
    en_empty_lines = len(df[df.iloc[:, 1] == ""])
    en_df = df["en"]
    es_df = df["es"]

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
            plt.text(i, v, f"{round(v / (en_empty_lines + len(en_df)) * 100, 2)}%")
        else:
            plt.text(i, v, f"{round(v / (es_empty_lines + len(es_df)) * 100, 2)}%")
    plt.xticks([1, 2], ["English", "Spanish"])
    leg = plt.legend(["Non-empty lines", "Empty lines"], loc="upper right")
    leg.legend_handles[0].set_color("blue")
    leg.legend_handles[1].set_color("red")
    plt.show()


def cluster_empty_lines(df: pd.DataFrame) -> pd.DataFrame:
    # iterate rows
    en_empty_mask = df["en"] == ""
    es_empty_mask = df["es"] == ""

    # Find rows where both English and Spanish lines are empty
    both_empty_mask = en_empty_mask & es_empty_mask

    # Find rows where only English lines are empty but not the previous Spanish line
    en_empty_prev_es_not_empty_mask = en_empty_mask & ~es_empty_mask.shift(fill_value=False)

    # Find rows where only Spanish lines are empty but not the previous English line
    es_empty_prev_en_not_empty_mask = es_empty_mask & ~en_empty_mask.shift(fill_value=False)

    # Combine masks to identify rows to remove
    rows_to_remove = both_empty_mask | en_empty_prev_es_not_empty_mask | es_empty_prev_en_not_empty_mask

    # Shift non-empty lines to fill in empty lines
    df["en"] = df["en"].shift(1).where(en_empty_mask, df["en"])
    df["es"] = df["es"].shift(1).where(es_empty_mask, df["es"])

    # Drop rows identified for removal
    df = df[~rows_to_remove]
    return df


def fix_empty_lines(df: pd.DataFrame) -> pd.DataFrame:

    # find row indexes where english lines are empty
    en_empty_lines = df[df.iloc[:, 0] == ""].index
    # for each line, if prev line has higher difference in number of words across languages, merge line and prev
    for i in en_empty_lines:
        prev_diff = abs(len(df.iloc[i-1, 0].split(" ")) - len(df.iloc[i-1, 1].split(" ")))
        next_diff = abs(len(df.iloc[i+1, 0].split(" ")) - len(df.iloc[i+1, 1].split(" ")))
        if prev_diff > next_diff:
            df.iloc[i-1, 0] += " " + df.iloc[i, 0]
            df.iloc[i-1, 1] += " " + df.iloc[i, 1]
            df.drop(i, inplace=True)
        else:
            df.iloc[i+1, 0] = df.iloc[i, 0] + " " + df.iloc[i+1, 0]
            df.iloc[i+1, 1] = df.iloc[i, 1] + " " + df.iloc[i+1, 1]
            df.drop(i, inplace=True)

    # find row indexes where spanish lines are empty
    es_empty_lines = df[df.iloc[:, 1] == ""].index
    # for each line, if prev line has higher difference in number of words across languages, merge line and prev
    for i in es_empty_lines:
        prev_diff = abs(len(df.iloc[i-1, 0].split(" ")) - len(df.iloc[i-1, 1].split(" ")))
        next_diff = abs(len(df.iloc[i+1, 0].split(" ")) - len(df.iloc[i+1, 1].split(" ")))
        if prev_diff > next_diff:
            df.iloc[i-1, 0] += " " + df.iloc[i, 0]
            df.iloc[i-1, 1] += " " + df.iloc[i, 1]
            df.drop(i, inplace=True)
        else:
            df.iloc[i+1, 0] = df.iloc[i, 0] + " " + df.iloc[i+1, 0]
            df.iloc[i+1, 1] = df.iloc[i, 1] + " " + df.iloc[i+1, 1]
            df.drop(i, inplace=True)

    return df


def print_empty_lines(df: pd.DataFrame):
    for i in range(len(df)):
        if df.iloc[i, 0] == "" or df.iloc[i, 1] == "":
            # print the entire row
            print(df.iloc[i, :])


if __name__ == "__main__":
    print("Hello from preprocessing.py")

    # make a single dataframe
    df = pd.read_csv("../europarl-extract-master/corpora/europarl.csv")

    # preprocess
    preprocessed_df = preprocess(df)
    #plot_empty_lines(preprocessed_df)
    len_preprocessed_df = len(preprocessed_df)
    assert len(preprocessed_df["en"]) == len(preprocessed_df["es"])
    print(f"Number of lines: {len(preprocessed_df['en'])}")
    preprocessed_df = cluster_empty_lines(preprocessed_df)
    len_clustered_df = len(preprocessed_df)
    print(f"Number of empty lines clustered: {len_preprocessed_df - len_clustered_df}")
    preprocessed_df = fix_empty_lines(preprocessed_df)
    print(f"Number of empty lines fixed: {len_clustered_df - len(preprocessed_df)}")
    print_empty_lines(preprocessed_df)

    # save the preprocessed data
    preprocessed_df.to_csv("../data/preprocessed_europarl.csv", index=False)