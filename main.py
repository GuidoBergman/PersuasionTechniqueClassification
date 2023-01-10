import argparse
from data import loader
from model_trainers.classifier import train_classifier

def tuple_arg(arg: str) -> tuple:
    arg_list = arg.split(",")
    return tuple(arg_list)


parser = argparse.ArgumentParser(prog="PMB Semantic Role Tagger",
                                 description="This program will classify VerbNet semantic roles for the PMB data")

parser.add_argument("--action", type=str,
                    choices=["load_data", "train_classifier"], help="The specific module to run", required=True)

parser.add_argument("--force_new", type=bool, default=False,
                    help="Force a redownload and regeneration of the source data")

parser.add_argument("--lang", type=tuple_arg, default="en",
                    help="The languages to use for the classification. Multiple languages can be specified with a comma in between (make sure not to include whitespace between entries)")

parser.add_argument("--qual", type=tuple_arg, default="gold,silver",
                    help="The quality partitions to use for the classification. Multiple partitions can be specified with a comma in between (make sure not to include whitespace between entries)")

if __name__ == "__main__":

    options = parser.parse_args()

    if options.action == "load_data":
        loader.load_data(options.lang, options.qual, options.force_new)
    elif options.action == "train_classifier":
        ds = loader.load_data(options.lang, options.qual, options.force_new)
        train_classifier(ds)
    else:
        print("no valid action selected!")
