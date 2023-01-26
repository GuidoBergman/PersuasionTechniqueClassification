
from datasets import Dataset

from sklearn.metrics import classification_report

from .torch_utils import class_vector_to_multi_hot_vector


def evaluate_model(ds: Dataset, inputs: list, labels: list, predictions: list) -> None:

    # for checking if role correct at index
    test_roles_ordered = []
    for i in ds["test"]:
        test_roles_ordered.append(i["verbnet"])

    label_list = ds["train"].features["verbnet"].feature.feature.names

    # check if predictions are correct at position
    # does not pick from the top ones
    for idx, pred in enumerate(predictions):
        pred_order = []
        for i in range(len(ds["test"][idx]["verbnet"])):
            possible_roles = [j for j in range(
                len(pred[i])) if pred[i][j] == 1]
            correct_pos = []
            for role in test_roles_ordered[idx][i]:
                if role in possible_roles:
                    correct_pos.append(True)
                else:
                    correct_pos.append(False)
            pred_order.append(correct_pos)

        print(f"original sentence: {inputs[idx]}")
        print(f"original labels: {ds['test'][idx]['verbnet']}")
        print(f"Predicted labels correct?: {pred_order}")

    token_labels = []
    token_preds = []

    for l, p in zip(labels, predictions):
        token_labels.extend(
            class_vector_to_multi_hot_vector(l, len(label_list)))
        token_preds.extend(
            class_vector_to_multi_hot_vector(p, len(label_list)))

    print(classification_report(
        y_true=token_labels, y_pred=token_preds, target_names=label_list, zero_division=0))
