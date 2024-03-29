import matplotlib.pyplot as plt
import pandas as pd


def og_info(lines: pd.DataFrame, filename: str):

    look_out = {}
    NaN_words = 0
    words = 0
    empty_lines_idx = []
    for line in lines:
        for word in line.split(' '):
            words += 1
            if word.isalnum() is False:
                if word in look_out:
                    look_out[word] += 1
                else:
                    look_out[word] = 1
        if line == "":
            empty_lines_idx.append(lines.index(line))

    # sort the dictionary by value
    look_out = dict(sorted(look_out.items(), key=lambda item: item[1], reverse=True))
    # save the dictionary to a file
    with open(f"../europarl-extract-master/corpora/{filename}_look_out.txt", "w") as file:
        file.write(str(look_out))

    # print empty line indexes
    unique_empty_lines_idx = list(set(empty_lines_idx))
    print(f"Empty lines found at indexes: {unique_empty_lines_idx}")

    # plot a barplot of the number of words, NaN words and non-alphanumeric words
    # in percentages
    plt.bar(["Alphanumeric words", "Non-alphanumeric words"],
            [(words - len(look_out))/words, len(look_out)/words])
    plt.title(f"Percentage of non-alphanumeric words in {filename}")
    plt.ylabel("Amount of words")
    # add percentages to the bars
    for i, v in enumerate([(words - NaN_words - len(look_out))/words, len(look_out)/words]):
        plt.text(i, v, f"{round(v*100, 2)}%")
    plt.show()


def plot_sentence_length(df:pd.DataFrame):
    en_df = df["en"]
    es_df = df["es"]
    # Count the number of words in each sentence
    en_lengths = [len(sentence.split(' ')) for sentence in en_df]
    es_lengths = [len(sentence.split(' ')) for sentence in es_df]
    # order lengths
    en_lengths.sort()
    es_lengths.sort()
    # print last 10 lengths
    print(en_lengths[-10:])  # [367, 367, 382, 407, 420, 450, 508, 633, 668, 668]
    print(es_lengths[-10:])  # [369, 374, 389, 393, 402, 423, 451, 511, 517, 658]
    # plot the lengths in a boxplot
    plt.boxplot([en_lengths, es_lengths])
    plt.title("Length of sentences in Europarl")
    plt.xlabel("Language")
    plt.xticks([1, 2], ["English", "Spanish"])
    plt.ylabel("Number of words")
    plt.show()


if __name__ == "__main__":
    # read the data
    df = pd.read_csv("../europarl-extract-master/corpora/europarl.csv")
    print(df.head())
    og_info(df["en"], "en_translation")
    og_info(df["es"], "es_translation")
    # plot the sentence length
    plot_sentence_length(df)
    print("done")
