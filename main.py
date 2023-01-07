import argparse
from data import loader

parser = argparse.ArgumentParser(prog="PMB Semantic Role Tagger",
                                 description="This program will classify VerbNet semantic roles for the PMB data")

parser.add_argument("--action", type=str, choices=["load_data"], help="The specific module to run", required=True)

parser.add_argument("--force_new", type=bool, default=False, help="Force a redownload and regeneration of the source data")

if __name__ == "__main__":

    options = parser.parse_args()

    if options.action == "load_data":
        loader.get_source(options.force_new)
    else:
        print("no valid action selected!")
