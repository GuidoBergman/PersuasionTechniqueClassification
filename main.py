import argparse
from data import loader
from model_trainers.classifier import train_classifier, evaluate_classifier


ACTIONS = [
    "load_data",
    "run_baseline",
    "train_classifier",
    "eval_classifier",
    "train_generator",
    "eval_generator"
]


def tuple_arg(arg: str) -> tuple:
    arg_list = arg.split(",")
    return tuple(arg_list)


parser = argparse.ArgumentParser(prog="Persuation techniques classification",
                                 description="This program will classify persuation techniques")

parser.add_argument("--action", type=str,
                    choices=ACTIONS, help="The specific module to run", required=True)

parser.add_argument("--force_new", action="store_true", default=False,
                    help="Force a redownload and regeneration of the source data")

parser.add_argument("--lang", type=tuple_arg, default="en",
                    help="The languages to use for the classification. Multiple languages can be specified with a comma in between (make sure not to include whitespace between entries)")

parser.add_argument("--qual", type=tuple_arg, default="gold,silver",
                    help="The quality partitions to use for the classification. Multiple partitions can be specified with a comma in between (make sure not to include whitespace between entries)")

parser.add_argument("--epochs", type=int, default=3,
                    help="The number of epochs the model will be trained")

parser.add_argument("--weighted", action="store_true", default=False,
                    help="Weigh the loss function according to the class distribution in the data (only for classifier)")

parser.add_argument("--name", type=str, default="model",
                    help="The name of the model to use")


if __name__ == "__main__":

    options = parser.parse_args()

    if options.action == "load_data":
        loader.load_data(options.lang, options.qual, options.force_new)
    elif options.action == "train_classifier":
        ds = loader.load_data(options.lang, options.qual, options.force_new)
        train_classifier(ds, options.name, options.epochs, options.weighted)
        evaluate_classifier(ds, options.name)
    elif options.action == "eval_classifier":
        ds = loader.load_data(options.lang, options.qual, options.force_new)
        evaluate_classifier(ds, options.name)


    else:
        print("no valid action selected!")
