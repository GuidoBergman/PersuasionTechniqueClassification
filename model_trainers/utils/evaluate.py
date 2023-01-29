
from datasets import Dataset

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from .torch_utils import class_vector_to_multi_hot_vector


def evaluate_model(ds: Dataset, inputs: list, labels: list, predictions: list) -> None:

    # for checking if role correct at index
    test_roles_ordered = []
    for i in ds["test"]:
        test_roles_ordered.append(i["verbnet"])

    label_list = ds["train"].features["verbnet"].feature.feature.names

    predicted_labels_role = []
    true_labels_role = []

    # check if predictions are correct at position
    # does not pick from the top ones
    for idx, pred in enumerate(predictions):
        pred_order = []
        for i in range(len(ds["test"][idx]["verbnet"])):
            true_labels_role.extend(ds["test"][idx]["verbnet"][i])
            possible_roles = pred[i]
            correct_pos = []
            for role in test_roles_ordered[idx][i]:
                if role in possible_roles:
                    correct_pos.append(True)
                    predicted_labels_role.append(role)
                else:
                    predicted_labels_role.append(0)
                    correct_pos.append(False)
            pred_order.append(correct_pos)

        # print(f"original sentence: {inputs[idx]}")
        # print(f"original labels: {ds['test'][idx]['verbnet']}")
        # print(f"Predicted labels correct?: {pred_order}")

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

    predicted_matrix = []
    true_matrix = []
    matrix_labels = ["0", "Agent", "Asset", "Attribute", "Co-Theme",
                     "Experiencer", "Location", "Theme", "PartOf", "Equal"]

    for idx, true in enumerate(true_labels_role):
        if true in matrix_labels:
            predicted_matrix.append(predicted_labels_role[idx])
            true_matrix.append(true_labels_role[idx])

    # true-pred matrix for certain labels
    matrix = confusion_matrix(
        true_matrix, predicted_matrix, labels=matrix_labels, normalize='true')
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=matrix_labels)

    # true-pred matrix for all labels
    # matrix = confusion_matrix(true_labels_role, predicted_labels_role, labels = label_list, normalize='true')
    # cm_display = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = label_list)

    cm_display.plot(cmap=plt.cm.YlGn)
    plt.xticks(rotation=90)
    #plt.savefig()
    plt.show()
