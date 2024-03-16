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
    with open(f"../data/{filename}_look_out.txt", "w") as file:
        file.write(str(look_out))
    # from the dictionary we learnt most words are alnum+punctuation
    # with the exception of - and longer hyphen
    # and '\xa0\xa0': 30828 instances in each file

    # print empty line indexes
    unique_empty_lines_idx = list(set(empty_lines_idx))
    print(f"Empty lines found at indexes: {unique_empty_lines_idx}")
    # idx 104 in en_translation and idx 438 in es_translation

    # plot a barplot of the number of words, NaN words and non-alphanumeric words
    # in percentages
    plt.bar(["Words", "Non-alphanumeric"],
            [(words - len(look_out))/words, len(look_out)/words])
    plt.title(f"Information about {filename}")
    plt.ylabel("Frequency")
    # add percentages to the bars
    for i, v in enumerate([(words - NaN_words - len(look_out))/words, len(look_out)/words]):
        plt.text(i, v, f"{round(v*100, 2)}%")
    plt.show()


def plot_sentence_length():
    # make a boxplot of the lengths of the sentences
    with open("../data/europarl-v7.es-en-en.txt") as file:
        en_df = file.read().split('\n')
    with open("../data/europarl-v7.es-en-es.txt") as file:
        es_df = file.read().split('\n')
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
    print("Hello from data_exploration.py")
    with open("../data/europarl-v7.es-en-en.txt") as file:
        en_df = file.read().split('\n')
    en_df = pd.DataFrame(en_df, columns=["en_translation"])
    with open("../data/europarl-v7.es-en-es.txt") as file:
        es_df = file.read().split('\n')
    es_df = pd.DataFrame(es_df, columns=["es_translation"])
    og_info(en_df, "en_translation")
    og_info(es_df, "es_translation")
    # plot_sentence_length()
    print("done")
