# NLP Final Assignment 

Files in order of use: 
1. `data_exploration.py` looks through both txt files and makes: 
    - `{filename}_look_out.txt` a dictionary with all non-alphanumeric words, ordered by frequency 
    - `{language}_percentage_alnum` a barplot showing the percentage of the data that is non-alphanumeric
    - `boxplot_len_sentences` a boxplot of the length of sentences in the data
    - prints the indexes of empty sentences in the data
    - prints the indexes of the 10 longest sentences of each language

2. `data_preprocessing.py` we do the following: 
    - remove non-alphanumeric characters from the beginning and end of sentences. Because there are so few honorific 
   titles and these don't consistently use a dot at the end (visible in look_out dictionary),
   we do not explicitly keep these.
    - get rid of words '\xa0\xa0', which appear 30828 times in each file
    - remove empty sentences
    - plot `empty_lines.png` which shows the amount of empty lines removed from each file
    - make `en_es_translation.csv` a csv file with the English and Spanish sentences in the same row, 
   with two columns "en_translation" and "es_translation"

3. `feat_processing.py` is a feature processing file:
   - eliminates the rows in which either of the sentences has more than 400 words (based on `boxplot_len_sentences` and
   print statements from `data_exploration.py`)
   - makes `after_boxplot_len_sentences.png` a new boxplot of the length of sentences in the data 
   - makes `cropped_en_es_translation.csv` a csv file without the outliers

## Seq2seq model
The seq2seq model is implemented in `seq2seq.py`.
The input to the model is generated during the processing steps outlined above.

`Lang` is a class that is used to create tokens for a language. 
The most important variables are `word2index` and `index2word`, which are dictionaries that map words to indices and vice versa. 
The class also has a `word2count` dictionary that keeps track of the frequency of each word in the data.

There are a few helper classes for conversions of words to tensors and indexes. These are: `indexesFromSentence`, `tensorFromSentence`, and `tensorsFromPair`. 
Their names are self-explanatory. 

`DataLoader` is a PyTorch class which nicely handles memory management for large datasets. By using the `DataLoader`, we hope not to exceed the GPU and CPU memory limits.