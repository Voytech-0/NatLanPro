# NLP Final Assignment 

## Data preprocessing
Files in order of use: 
1. We use the europarl extractor (more information in `europarl-extract-master/README.md`) to do:
   - `cleanSourceFiles.sh` to remove XML markup and empty lines
   - `disambiguate_speaker_IDs.py` to disambiguate statement IDs
   - `segment_EuroParl.sh` to segment the sentences
   - `segment-tokenise_EuroParl.sh` to segment and tokenise the sentences
   - `segment-tokenise_ixaPipes.sh` to segment and tokenise the sentences with ixa-pipe-tok
   - `extract_corpora.sh` to extract the corpora 
   - results in the `corpora/europarl.csv` file

2. `data_exploration.py` looks through the resulting data: 
    - `{filename}_look_out.txt` a dictionary with all non-alphanumeric words, ordered by frequency 
    - `{language}_percentage_alnum` a barplot showing the percentage of the data that is non-alphanumeric
    - `boxplot_len_sentences` a boxplot of the length of sentences in the data
    - prints the indexes of empty sentences in the data
    - prints the indexes of the 10 longest sentences of each language

3. `preprocessing.py` we do the following: 
    - remove non-alphanumeric characters from the beginning and end of sentences. Because there are so few honorific 
   titles and these don't consistently use a dot at the end (visible in look_out dictionary),
   we do not explicitly keep these.
    - cluster half-empty rows (rows with only one and same language present), happens to around 10 000 rows
    - look at the previous and next row to see which one is missing more words. Half-empty rows merge with that one. 
   This only happens 18 times across the data and they have been checked manually by a native speaker.
    - plot `empty_lines.png` which shows the amount of empty lines removed from each file
    - make `preprocessed_europarl.csv` a csv file with the English and Spanish sentences in the same row, with two columns "en" and "es"

4. `feat_processing.py` is a feature processing file:
   - eliminates the rows in which either of the sentences has more than 50 words (based on `boxplot_len_sentences` and
   print statements from `exploration.py`)
   - makes `after_boxplot_len_sentences.png` a new boxplot of the length of sentences in the data 
   - makes `cropped_europarl.csv` a csv file without the outliers

## Seq2seq model
The seq2seq model is implemented in `seq2seq.py`.
The input to the model is generated during the processing steps outlined above.

`Lang` is a class that is used to create tokens for a language. 
The most important variables are `word2index` and `index2word`, which are dictionaries that map words to indices and vice versa. 
The class also has a `word2count` dictionary that keeps track of the frequency of each word in the data.

There are a few helper classes for conversions of words to tensors and indexes. These are: `indexesFromSentence`, `tensorFromSentence`, and `tensorsFromPair`. 
Their names are self-explanatory. 

`DataLoader` is a PyTorch class which nicely handles memory management for large datasets. By using the `DataLoader`, we hope not to exceed the GPU and CPU memory limits.
