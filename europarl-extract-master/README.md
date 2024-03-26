# EuroparlExtract

EuroparlExtract is a toolkit for the extraction of parallel and comparable corpora from the Europarl corpus.

## Dependencies

EuroparlExtract comprises a number of scripts written in Python3 and Bash for Linux; Windows is not yet supported. 
Some parts of the package are based on third-party software (more details in original repository).
All dependencies have been added to the `requirements.txt` file.

The following step-by-step instructions will guide you through the corpus extraction process.

## Preprocess Europarl Source Files

The original Europarl source files need to be cleaned and normalised before applying the corpus extraction scripts. 

# VERY IMPORTANT
You need to download `europarl_statements.csv` from the corpora folder of the original repository and add it to the corpora folder of this repository.
We add our txt files `europarl-v7.es-en-en.txt` and `europarl-v7.es-en-es.txt` to the corpora folder.
We execute all commands from the `europarl-extract-master` directory

To perform the required preprocessing steps, you can **either** follow the preprocessing steps 1 to 3 **or** execute 
`./preprocess/preprocess_batch.sh corpora` and then proceed directly to [Extract Corpora](#extract-corpora).

### 1. Remove XML Markup and Empty Lines

First, remove spurious XML markup, empty lines etc. with the supplied bash script `cleanSourceFiles.sh <input_folder>`, e.g.:

```shell
./preprocess/cleanSourceFiles.sh corpora
```

### 2. Disambiguate Statement IDs

Next, run the script `disambiguate_speaker_IDs.py <input_folder>` to ensure that no two statements are assigned the same ID within one source file. To do so, run:

```shell
python3 disambiguate_speaker_IDs.py corpora
```

### 3. Sentence Segmentation and Optional Tokenisation

For the extraction of **sentence-aligned parallel corpora, sentence segmentation is a required** pre-processing step, whereas in the case of comparable corpora sentence segmentation is not required (albeit useful for future analyses). Tokenisation is optional for both comparable and parallel corpora and therefore depends on end users' needs.

EuroparlExtract comes with **two different third-party tools** users can choose from: 1) *ixa-pipe-tok*, a sentence splitter and tokeniser implemented in Java; or 2) the sentence splitter and tokeniser of the *Europarl Preprocessing Tools* implemented in Perl. The former is more accurate but considerably slower that the latter, so users should choose one of the tools according to their own preferences.

To perform sentence segmentation without tokenisation using *Europarl Preprocessing Tools*, run:

```shell
./preprocess/segment_EuroParl.sh corpora
```

For segmentation and tokenisation using *Europarl Preprocessing Tools*, run:

```shell
./preprocess/segment-tokenise_EuroParl.sh corpora
```

For segmentation and subsequent tokenisation using *ixa-pipe-tok*, run:

```shell
./preprocess/segment-tokenise_ixaPipes.sh corpora
```

**Notes:**
- You only need to choose one of the three methods above!
- You may use your own/other tools for sentence segmentation and tokenisation. If you choose to do so, make sure that segmented/tokenised files are files of the type `.txt` and that XML markup is retained.
- When using *Europarl Preprocessing Tools*, you may first only segment the source files and tokenise them later.
- Running *ixa-pipe-tok* requires Java 1.7+ on your system. You can install it with `sudo apt-get install openjdk-8-jdk`.


## Extract Corpora

After preprocessing the Europarl source files, the extraction can be performed by calling the main script `extract.py` 
with either the `parallel` or `comparable` subcommand. In our case we use comparable, since is between two individual languages.


### b) Comparable Corpora

Contrary to parallel corpora, comparable corpora consist of individual monolingual files in the choosen language(s) rather than of bilingual source-target text pairs. In comparable corpora, two sections can be distinguished: one containing only texts originally produced in a given language (e.g. non-translated English), and one containing only texts that have been translated into a given language (e.g. translated English). The latter can be further subdivided according to source languages (e.g. English texts translated from Polish, English texts translated from German ...). Note that no source texts are stored in the translated section of comparable corpora, i.e. only the target side of each language combination is extracted, while source language information is only used as metadata. To extract comparable corpora, the following arguments need to be specified (see also help message `python3 extract.py comparable --help`):

- `-sl [source_language ...]`: Choose one or more source language(s), separated by blanks. For a list of supported language codes, display the help message by calling `python3 extract.py comparable --help`. Note: you may also choose `all` source languages.
- `-tl [target_language ...]`: Choose one or more target language(s), separated by blanks. For a list of supported languages, display the help message by calling `python3 extract.py comparable --help`. Note: you may also choose `all` target languages.
- `-i <input_folder>`:  Path to input folder containing Europarl source files, usually txt/.
- `-o <output_folder>`: Path to output folder where subfolders for each language pair will be created.
- `-s <statement_file>`: Optional argument to supply a precompiled statement list (CSV format) rather than creating the list from Europarl source files from scratch (**recommended** - extremly speeds up the extraction process!) The list can be found in the folder [corpora/](https://github.com/mustaszewski/europarl-extract/tree/master/corpora) of the EuroparlExtract distribution.
- `-al`: Optional argument to disseminate parenthesised language tags across source files (**recommended** - largely increases number of extractable statements!)
- `-c {lang|speaker|both}`: Optional argument to remove parenthesised language identifiers and/or speaker metadata tags from output files.
- `-d`: Optional argument to create a log file for debugging (not recommended - use only in case of problems).

**Example:**

```shell
python3 extract.py comparable -sl all -tl PL BG -i corpora/ -o corpora/ -s corpora/europarl_statements.csv -al -c speaker
```
Extracts texts originally written in Polish and texts originally written in Bulgarian, as well as texts translated into 
these two languages from all other Europarl languages; speaker metadata markup removed from output files.

And finally add the files in a csv for further processing:
    
```shell
python3 make_csv.py
```

# Further information

## Performance
The script extract.py is not speed-optimised. Therfore, the first part of the extraction step may take several hours, depending on the CPU used. However, the proces can be speeded up extremely if the precompiled list of Europarl statements (see corpora/ folder of this package) is provided to the script. To do so, specify the path of the list via the `-s` parameter. Using the precompiled list, the extraction of the corpora of your choice should take only between a few minutes and up to one hour, depending on your CPU and the amount of text to be extracted. 


## Citation

If you use EuroparlExtract or the corpora derived from it in your work, please cite the following paper (open access):

>
>Ustaszewski, Michael (2018): Optimising the Europarl corpus for translation studies with the EuroparlExtract toolkit. In: *Perspectives - Studies in Translation Theory and Practice* 27:1, p. 107-123. DOI: [10.1080/0907676X.2018.1485716](https://doi.org/10.1080/0907676X.2018.1485716)
>

## Third-party software

EuroparlExtract uses the following third-party software:

* A customisation of [ixa-pipe-tok](https://github.com/ixa-ehu/ixa-pipe-tok) by Rodrigo Agerri for sentence splitting and tokenisation.
* The [Europarl Preprocessing Tools](http://www.statmt.org/europarl) by Philipp Koehn for sentence splitting and tokenisation.
* A customisation of [GaChalign](https://github.com/alvations/gachalign) by Liling Tan and Francis Bond for sentence alignment.


## Credits

* [Europarl-extract-master](https://github.com/mustaszewski/europarl-extract) original repository by Michael Ustaszewski

