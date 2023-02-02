import json
import os
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
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
                role_frequency[role] +=1

    print(len(dict))
    with open('data/lookup_table.json', 'w') as fp:
        json.dump(dict, fp, sort_keys=True, indent=4)

    with open('data/frequency_overview.json', 'w') as fp:
        json.dump(role_frequency, fp, sort_keys=True, indent=4)



# def jsonKV2int(x):
#     if isinstance(x, dict):
#             return {k:(int(v) if isinstance(v, unicode) else v) for k,v in x.items()}
#     return x
def evaluate_baseline(ds):
    with open('data/lookup_table.json') as json_file:

        # dict = json.load(json_file, object_hook=jsonKV2int)
        dict = json.load(json_file)

    print(len(dict))

    label_list = ds["train"].features["verbnet"].feature.feature.names

    roles = {}
    for label in label_list:
        roles[label] = 0

    correct = 0
    incorrect = 0
    nr_words = 0

    # df = {'key1': 5, 'key2': 35, 'key3': 25}
    #
    # max_value = max(df, key=df.get)
    #
    # print("Maximum value = ", max_value)
    # print("Output1:", sorted(df.values())[-1])




    for i, test_sentence in enumerate(ds["test"]):
        # if i > 2:
        #     return

        print(i, "/", len(ds["test"]))


        for idx, word in enumerate(test_sentence["tok"]):
            word = porter_stemmer(lemmatizer(word))
            # print(type(dict[word]["Agent"]))
            nr_words += 1
            role_nr_correct = 0
            role_nr_expected = len(test_sentence["verbnet"][idx])
            for role_id in test_sentence["verbnet"][idx]:
                if word not in dict:
                    prediction = label_list[0]
                if word in dict:
                    # for role in dict[word]:
                    #     dict[word][role] = int(dict[word][role])
                    # print(type(dict[word]["Agent"]))
                    prediction = max(dict[word], key=dict[word].get)
                    df = dict[word]
                    # print(df)
                    # prediction = sorted(df.values())[-1]
                    # print(sorted(df.values())[-1])
                # print(word, "pred= ", prediction, "gold st=", label_list[role_id])
                if prediction == label_list[role_id]:
                    role_nr_correct += 1
            if role_nr_correct == role_nr_expected:
                # print("correct")
                correct += 1
            else:
                incorrect += 1

    print(len(test_sentence["tok"]))
    print(correct, incorrect)
    print(nr_words)

    # print(test_roles_ordered)






    # n = 0
    # for idx, word in enumerate(ds["train"][n]["tok"]):
    #     print(word, ds["train"][n]["verbnet"][idx])
    #     for roleid in ds["train"][n]["verbnet"][idx]:
    #         print(label_list[roleid])

    # f.close()