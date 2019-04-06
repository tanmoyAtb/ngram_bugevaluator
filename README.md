# ngram_bugevaluator
A skip-gram model of word embeddings to learn code structures and evaluate buggy codes

Clone this repo, cd into the repo and make sure python3 and pip is installed.
Run the following commands

```
python3 -m venv env
source env/bin/activate
pip install -r plugins.txt
python main.py
```

# Details
A tensorflow skip-gram model to train word embeddings taking a line of code as token.
The IntroClass Benchmark set of codes was used. 
The model has been trained using correct codes of smallest, grade, syllables, median and checksum.
The validation set was built by using incremented mutation in the correct codes.

A prob distribution was calculated using word embeddings from training on the validation set.
The results show a decrease in prob with increase in mutation.

The process can be used as an indicator of naturalness of code.
