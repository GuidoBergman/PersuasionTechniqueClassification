# Comparing rule-based and neural approaches to Semantic Role Labeling

Semantic Role Labelling (SRL) is an important part of many Natural Language Processing tasks. This study examines two approaches, a classification model and a generative sequence-to-sequence (seq2seq) model, compared to a classical rule-based approach. We use data from the [Parallel Meaning Bank](https://pmb.let.rug.nl/), which has the added challenge that a single sentence part can have multiple, ordered roles. The results highlight that all approaches have advantages and disadvantages, indicating that a combined approach for SRL might be advisable.

This repository is part of a course project for the course Computational Semantics of the [University of Groningen](https://www.rug.nl/).

## Installation

> :warning: A Python version of at least 3.9 is recommended to run this script

It is strongly recommended to create a new virtual environment before installation of any dependencies. This can be done with the following command:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

All dependecies needed to run the module can be installed by running the following command:

```bash
pip3 install -r requirements.txt
```

To install dev-only requirements as well, you can additionally run `pip3 install -r requirements-dev.txt`

## Running the code

All experiments can be started and controlled from the command line. Simply start the main file with the specified option by running the following command:

```bash
python3 main.py --action train_classifier
```

The available actions are:

- `load_data`: Load and preprocess the data from the PMB for further processing
- `run_baseline`: Train and evaluate the baseline model
- `train_classifier`: Train and evaluate the classifier model
- `eval_classifier`: Evaluate an existing classifier model
- `train_generator`: Train and evaluate the seq2seq model
- `eval_generator`: Evaluate an existing seq2seq model

Additionally, the following options can be specified to further configure the program:

|Option|Description|Default Value|
|------|-----------|-------------|
|`lang`|The language partitions of the PMB to use |en|
|`qual`|The quality splits of the PMB data which will be used| gold,silver |
|`version`|The version of the DBpedia infobox dump to use for analysis|2022.03.01|
|`epochs`|The number of epochs to train the models for|3|
|`weighted`|Wether the loss function of the classifier should be weighted according to the class distribution in the data |False|
|`force_new`|Force a regeneration of the PMB data and all processed datasets|False|
