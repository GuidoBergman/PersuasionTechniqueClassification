
from datasets import Dataset

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from .torch_utils import class_vector_to_multi_hot_vector


def evaluate_model(ds: Dataset, inputs: list, labels: list, predictions: list) -> None:

    # for checking if role correct at index
    test_roles_ordered = []
    for i in ds["test"]:
        test_roles_ordered.append(i["verbnet"])

    label_list = ds["train"].features["verbnet"].feature.feature.names

    # for confusion matrix
    predicted_labels_role = []
    true_labels_role = []

    # for bar chart of percentage of missed/excess args
    role_lengths = []

    # check if predictions are correct at position
    # does not pick from the top ones
    for idx, pred in enumerate(predictions):
        pred_order = []
        for i in range(len(ds["test"][idx]["verbnet"])):
            true_labels_role.extend(ds["test"][idx]["verbnet"][i])
            possible_roles = pred[i]
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
        # print(f"original labels: {ds['test'][idx]['verbnet']}")
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

    predicted_matrix1, predicted_matrix2 = [], []
    true_matrix1, true_matrix2 = [], []
    # matrix_labels = ["0", "Agent", "Attribute", "Co-Theme", "Destination",
    #                  "Experiencer", "Location", "Patient", "Theme", "PartOf", "Equal"]
    matrix_labels_1 = ["0", "Agent", "Attribute",
                       "Experiencer", "Location", "Theme"]
    matrix_labels_2 = ["0", "InstanceOf", "Sub",
                       "Material", "Extent", "Co-Patient"]
    for idx, true in enumerate(true_labels_role):
        if true in matrix_labels_1:
            predicted_matrix1.append(predicted_labels_role[idx])
            true_matrix1.append(true_labels_role[idx])
        if true in matrix_labels_2:
            predicted_matrix2.append(predicted_labels_role[idx])
            true_matrix2.append(true_labels_role[idx])

    # true-pred confusion matrix for high freq
    matrix_1 = confusion_matrix(
        true_matrix1, predicted_matrix1, labels=matrix_labels_1, normalize='true')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix_1, display_labels=matrix_labels_1)

    plt.rcParams.update(
        {'axes.labelsize': 20, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
    cm_display.plot(cmap=plt.cm.YlGn)
    plt.xticks(rotation=90)
    plt.savefig("confusion_matrix_high_freq.png",format="png")

    # true-pred confusion matrix for low freq
    matrix_2 = confusion_matrix(
        true_matrix2, predicted_matrix2, labels=matrix_labels_2, normalize='true')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix_2, display_labels=matrix_labels_2)
    cm_display.plot(cmap=plt.cm.YlGn)
    plt.xticks(rotation=90)
    plt.savefig("confusion_matrix_low_freq.png",format="png")

    # true-pred confusion matrix for all labels
    matrix_3 = confusion_matrix(
        true_labels_role, predicted_labels_role, labels=label_list, normalize='true')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix_3, display_labels=label_list)
    cm_display.plot(cmap=plt.cm.YlGn)
    plt.xticks(rotation=90)
    plt.savefig("confusion_matrix_all.png",format="png")

    # bar chart of percentage of missed/excess args
    labels = ['1', '2', '3+']
    print(f"role lengths: {role_lengths}")
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

    excess = [perc_excess_1, perc_excess_2, perc_excess_3ab]
    missing = [perc_missing_1, perc_missing_2, perc_missing_3ab]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, missing, width, label='Missed')
    rects2 = ax.bar(x + width/2, excess, width, label='Excess')
    ax.set_ylabel("% of tokens")
    ax.set_ylabel("Number of tokens")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig("excess_args_bar.png",format="png")
