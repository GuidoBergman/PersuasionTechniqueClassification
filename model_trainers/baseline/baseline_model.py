import json
import os
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# pylint: disable-next=relative-beyond-top-level
from ..utils import evaluate_model

# uncomment these if you want to run the training
# nltk.download('wordnet')
# nltk.download('punkt')


def lemmatizer(token):

    wordnet_lemmatizer = WordNetLemmatizer()
    if token == "'m":
        return "be"
    return wordnet_lemmatizer.lemmatize(token, pos="v")


def lancaster_stemmer(token):
    lancaster = LancasterStemmer()
    return lancaster.stem(token)


def porter_stemmer(token):
    porter = PorterStemmer()
    return porter.stem(token)


def run_baseline(ds):

    if os.path.isfile("data/lookup_table.json") and os.path.isfile("data/frequency_overview.json"):
        print("## Lookup table already exists ##")
        return

    label_list = ds["train"].features["verbnet"].feature.feature.names

    roles = {}
    for label in label_list:
        roles[label] = 0

    role_frequency = roles.copy()
    dict = {}

    for i, train_sentence in enumerate(ds["train"]):

        print(i, "/", len(ds["train"]))

        for idx, word in enumerate(train_sentence["tok"]):
            word = porter_stemmer(lemmatizer(word))
            for role_id in train_sentence["verbnet"][idx]:
                if word not in dict:
                    dict[word] = roles.copy()
                role = label_list[role_id]
                dict[word][role] += 1
                role_frequency[role] += 1

    print(len(dict))
    with open('data/lookup_table.json', 'w') as fp:
        json.dump(dict, fp, sort_keys=True, indent=4)

    with open('data/frequency_overview.json', 'w') as fp:
        json.dump(role_frequency, fp, sort_keys=True, indent=4)


def evaluate_baseline(ds):

    # Loads the lookup table
    with open('data/lookup_table.json') as json_file:
        dict = json.load(json_file)

    # Loads the general overview of all the classes and their frequency
    with open('data/frequency_overview.json') as json_file:
        frequency_table = json.load(json_file)

    # retrieves a list of all the labels in the PBM
    label_list = ds["train"].features["verbnet"].feature.feature.names

    # creates empty lists that are needed for evaluation
    predictions = []
    labels = []
    input = []

    # goes through all test sentences
    for i, test_sentence in enumerate(ds["test"]):
        # print(i, "/", len(ds["test"]))

        # creates temporary lists per sentence
        prediction = []
        label = []

        # goes through all the words in a sentence
        for idx, word in enumerate(test_sentence["tok"]):
            input.append(word)

            # use the lemmatized version
            word = porter_stemmer(lemmatizer(word))

            # if the word is not in the lookup table, use the normal frequency table
            if word not in dict:
                word_dict = frequency_table.copy()
            else:
                word_dict = dict[word].copy()

            # make temporary lists for each label for a word
            tm_labels = []
            tm_predictions = []
            for role_id in test_sentence["verbnet"][idx]:

                # prediction is the role with the highest frequency
                pred = max(word_dict, key=word_dict.get)

                # 1st prediction gets set to 0 to ensure that 2nd prediction uses 2nd highest frequency
                word_dict[pred] = 0
                tm_predictions.append(label_list.index(pred))
                tm_labels.append(role_id)
            prediction.append(tm_predictions)
            label.append(tm_labels)
        predictions.append(prediction)
        labels.append(label)

    print("## finished testing ##")
    print("## starting evaluation ##")
    evaluate_model(ds, input, labels, predictions)
