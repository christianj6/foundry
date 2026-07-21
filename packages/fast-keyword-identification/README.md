### Fast Keyword Identification
Fast keyword identification with n-gram vector string matching.
***

#### Overview
This package provides a generic pipeline for fuzzy-identification of keywords in large document collections. For example, if you wish to find all occurrences of the keyword "Walmart" in a large document collection, but expect some typos or variations in spelling, this module will allow you to quickly identify all matches. The matcher is based on a character n-gram vector model rather than the slower string edit distance. The module is originally intended for brand monitoring applications.

***

### Installation

```
pip install fast-keywords
```

***

### CLI

```
python -m fast_keywords --help
```

***

### Usage

```
python -m fast_keywords -k keywords.txt -c corpus.csv -l english -b 0.75
```

***

### Training Models for Additional Filtering

While the main script will search for keywords in the provided corpus, filtering according to the match confidence, you can also train and use simple text classifiers as an additional filter to remove dubious matches. For example, if you are searching for the company "apple," but find your searches frequently return references to fruit, you can train a model which will exclude those matches based on the surrounding text of matched keywords. Instructions for model training and usage are provided below.

1. After searching for keywords you will find a column "Match is Invalid" in the ```output.xlsx``` file.

2. Modify this column, changing matches which should be filtered out to "1".

3. Train a new model using the ```--train``` flag, providing the modified ```output.xlsx``` file and the original keywords file, as in the command below.

   1. ```
      python -m fast_keywords --train -d output.xlsx -k keywords.txt
      ```

4. The train command will create a directory with several ```model.pb``` files which you can distribute and use for filtering. You should use the absolute path to this containing directory as the model path passed with the ```-m``` flag.

5. You can use your models when predicting as in the below command. You can also pass previously-trained models using the ```-m``` flag to continue training on new data when running the train command.

```
python -m fast_keywords -k keywords.txt -c corpus.csv -l english -b 0.75 -m model.pb
```

***

### Notes

- Your input .csv must have a "text" column containing documents.
- The main script will create a a file ```output.xlsx``` summarizing identified keywords and their metadata.
