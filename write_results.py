import csv
import json
import os
import pickle
import shutil
import time
from os import path

from global_parameters import print_message, GlobalParameters
from new_xlsx_file import new_write_file_content

global method_names


def print_results(results):
    for config in results.keys():
        current_key = json.loads(config)
        print("-----------------")
        print(json.dumps(current_key, indent=4, sort_keys=True))
        print(json.dumps(results[config], indent=4, sort_keys=True))
        print("-----------------")


def add_to_csv(results, result_path):
    path = os.path.join(result_path, 'results.csv')

    if os.path.exists(path):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    values = []
    for measure, value in results.items():
        header = [""]
        values.append([value["config"]["features"]])
        for model, result in value["config"]["results"].items():
            header.append(model)
            values[-1].append(result)

    with open(path, append_write) as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if append_write == 'w':
            wr.writerow(header)
        wr.writerows(values)




def write_results(results):
    glbs = GlobalParameters()
    print_message("Writing results...")
    # add_to_csv(results, glbs.RESULTS_PATH)

    pickle_path = os.path.join(glbs.RESULTS_PATH, "Pickle files")
    if path.exists(pickle_path):
        shutil.rmtree(pickle_path, ignore_errors=True)
    os.makedirs(pickle_path)

    xlsx_path = os.path.join(glbs.RESULTS_PATH, "Xlsx files")
    if path.exists(xlsx_path):
        shutil.rmtree(xlsx_path, ignore_errors=True)
    time.sleep(0.5)
    os.makedirs(xlsx_path)

    for key in results.keys():
        with open(os.path.join(pickle_path, key) + ".pickle", "wb+") as file:
            pickle.dump(results[key], file)
        new_write_file_content(os.path.join(pickle_path, key) + ".pickle", key, xlsx_path)
