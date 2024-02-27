
from datasets import Dataset

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from .torch_utils import class_vector_to_multi_hot_vector
from typing import Optional

from ..classifier.models import LABEL_LIST

def evaluate_model(ds: Dataset, inputs: list, labels: list, predictions: list, output_length: Optional[list] = None) -> None:

    # for checking if role correct at index
    test_roles_ordered = []
    for i in ds["dev"]:
        test_roles_ordered.append(i["Techniques"])

    label_list = ['No_Technique'] + LABEL_LIST

    # for confusion matrix
    predicted_labels_role = []
    true_labels_role = []

    # for bar chart of percentage of missed/excess args
    role_lengths = []

    if output_length is not None:
        short_sent = list(filter(lambda e: e == "short", output_length))
        long_sent = list(filter(lambda e: e == "long", output_length))

        long_sent_freq = len(long_sent)/len(output_length)
        short_sent_freq = len(short_sent)/len(output_length)

        print(f"too long sequences: {round(long_sent_freq, 4)}")
        print(f"too short sequences: {round(short_sent_freq, 4)}")

    # check if predictions are correct at position
    # does not pick from the top ones
    for idx, pred in enumerate(predictions):
        pred_order = []
        for i in range(len(ds["dev"][idx]["Techniques"])):
            true_labels_role.extend(ds["dev"][idx]["Techniques"][i])
            try:
                possible_roles = pred[i]
            except IndexError:
                print("'### Alignment Error ###")
                print(pred)
                print(ds["dev"][idx]["Techniques"])
                continue
            correct_pos = []
            # print(f"possible roles for {test_roles_ordered[idx][i]}: {possible_roles}")
            for j, role in enumerate(test_roles_ordered[idx][i]):
                # if theres missing roles in predictions add 0
                if j >= len(possible_roles):
                    predicted_labels_role.append(0)
                    correct_pos.append(False)
                # check if true role at current position matches
                # predicted role at curr position
                elif role == possible_roles[j]:
                    correct_pos.append(True)
                    predicted_labels_role.append(role)
                else:
                    predicted_labels_role.append(possible_roles[j])
                    correct_pos.append(False)
            role_lengths.append(len(possible_roles) -
                                len(test_roles_ordered[idx][i]))
            pred_order.append(correct_pos)

        # print(f"original sentence: {inputs[idx]}")
        # print(f"original labels: {ds['dev'][idx]['Techniques']}")
        # print(f"Predicted labels correct?: {pred_order}")
        # print(f"excess or misssing: {role_lengths}")

    token_labels = []
    token_preds = []

    for l, p in zip(labels, predictions):
        token_labels.extend(
            class_vector_to_multi_hot_vector(l, len(label_list)))
        token_preds.extend(
            class_vector_to_multi_hot_vector(p, len(label_list)))

    print(classification_report(
        y_true=token_labels, y_pred=token_preds, target_names=label_list, zero_division=0))

    predicted_labels_role = [label_list[j] for j in predicted_labels_role]
    true_labels_role = [label_list[j] for j in true_labels_role]


    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true=true_labels_role, y_pred=predicted_labels_role, labels=label_list, normalize="true", xticks_rotation="vertical", cmap=plt.cm.YlGn, values_format=".1f")
    plt.rcParams.update(
        {"axes.labelsize": 20, "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig = cm_display.figure_
    fig.set_figwidth(25)
    fig.set_figheight(25)
    plt.savefig("confusion_matrix_all.png", format="png",
                dpi=600, bbox_inches="tight")

    # bar chart of percentage of missed/excess args
    labels = ["1", "2", "3+"]
    perc_excess_1 = (
        len([y for y in role_lengths if y == 1])/len(role_lengths)) * 100
    perc_excess_2 = (
        len([y for y in role_lengths if y == 2])/len(role_lengths)) * 100
    perc_excess_3ab = (
        len([y for y in role_lengths if y >= 3])/len(role_lengths)) * 100
    perc_missing_1 = (
        len([y for y in role_lengths if y == -1])/len(role_lengths)) * 100
    perc_missing_2 = (
        len([y for y in role_lengths if y == -2])/len(role_lengths)) * 100
    perc_missing_3ab = (
        len([y for y in role_lengths if y <= -3])/len(role_lengths)) * 100

    excess = [round(perc_excess_1, 2), round(
        perc_excess_2, 2), round(perc_excess_3ab, 2)]
    missing = [round(perc_missing_1, 2), round(
        perc_missing_2, 2), round(perc_missing_3ab, 2)]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, missing, width, label="Missed")
    rects2 = ax.bar(x + width/2, excess, width, label="Excess")
    ax.set_ylabel("% of tokens")
    ax.set_xlabel("Number of tokens")
    ax.set_ylim(ymin=0, ymax=100)
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.savefig("excess_args_bar.png", format="png",
                dpi=300, bbox_inches="tight")
