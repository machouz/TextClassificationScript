import json
import os
import shutil
import traceback
import sys

from classification import classify
from feature_selction import get_selected_features
from features import extract_features
from global_parameters import GlobalParameters, print_message
from normalization import normalize
from notification_handler import send_work_done
from stylistic_features import text_language
from words_cloud import generate_word_clouds
from write_results import write_results


def set_global_parameters(configs):
    glbls = GlobalParameters()
    config = configs[1]
    glbls.FILE_NAME = configs[0]
    glbls.FEATURES = config["features"]
    glbls.NORMALIZATION = "".join(sorted(config["nargs"].upper()))
    glbls.OUTPUT_DIR = config["output_csv"]
    glbls.METHODS = config["methods"]
    glbls.TRAIN_DIR = config["train"]
    glbls.TEST_DIR = config["test"]
    glbls.RESULTS_PATH = config["results"]
    glbls.MEASURE = config["measure"]
    glbls.STYLISTIC_FEATURES = config["stylistic_features"]
    glbls.SELECTION = config["selection"].items() if "selection" in config else []
    try:
        if 'language' in config:
            glbls.LANGUAGE = config['language']
        else:
            path = os.path.join(config["train"], os.listdir(config["train"])[0])
            glbls.LANGUAGE = text_language(open(path, "r", encoding="utf8", errors='replace').read())
    except:
        glbls.LANGUAGE = "english"


def print_run_details():
    glbs = GlobalParameters()
    print(
        """
	---------------------------------------
	Training path: {}
	Testing Path: {}
	Features: {}
	Stylistic Features: {}
	Normalization: {}
	Methods: {}
	Measure: {}
	Output Path: {}
	Results Path: {}
	---------------------------------------
	""".format(
            glbs.TRAIN_DIR,
            glbs.TEST_DIR,
            glbs.FEATURES,
            glbs.STYLISTIC_FEATURES,
            glbs.NORMALIZATION,
            glbs.METHODS,
            glbs.MEASURE,
            glbs.OUTPUT_DIR,
            glbs.RESULTS_PATH,
        )
    )


def add_results(old_results):
    glbs = GlobalParameters()
    temp = {}
    temp["results"] = old_results[glbs.FILE_NAME]
    temp["featurs"] = glbs.FEATURES
    temp["normalization"] = glbs.NORMALIZATION
    temp["stylistic_features"] = glbs.STYLISTIC_FEATURES
    old_results[glbs.FILE_NAME] = temp
    return old_results


def divide_results(result):
    new_result = {
        "accuracy_score": {},
        "f1_score": {},
        "precision_score": {},
        "recall_score": {},
        "roc_auc_score": {},
        "confusion_matrix": {},
        "roc_curve": {},
        "precision_recall_curve": {},
        "accuracy_&_confusion_matrix": {},
    }

    for config_name, dic in result.items():
        for method, score in dic["results"].items():
            for measure, value in score.items():
                new_result[measure][config_name] = {}
                new_result[measure][config_name]["results"] = {}

    for config_name, dic in result.items():
        for method, score in dic["results"].items():
            for measure, value in score.items():
                new_result[measure][config_name]["results"][method] = value
                new_result[measure][config_name]["featurs"] = dic["featurs"]
                new_result[measure][config_name]["normalization"] = dic["normalization"]
                new_result[measure][config_name]["stylistic_features"] = dic[
                    "stylistic_features"
                ]

    return_results = {}
    for measure, data in new_result.items():
        if new_result[measure] != {}:
            return_results[measure] = data

    return return_results


def main(cfg):
    try:
        glbs = GlobalParameters()
        configs = get_cfg_files(cfg)
        results = {}
        n_test_dir = ""
        total_files = len(configs)
        for i, config in enumerate(configs):
            print_message("Running config {}/{}".format(i + 1, total_files))
            set_global_parameters(config)
            print_run_details()
            n_train_dir = normalize()
            if glbs.TEST_DIR != "":
                n_test_dir = normalize(test=True)
            train, tr_labels, test, ts_labels, all_features = extract_features(
                n_train_dir, n_test_dir
            )
            for selection in glbs.SELECTION:
                try:
                    train, test = get_selected_features(
                        selection, train, tr_labels, test, ts_labels, all_features
                    )
                except:
                    pass
            results[glbs.FILE_NAME] = classify(
                train, tr_labels, test, ts_labels, all_features
            )
            results = add_results(results)
        if glbs.WORDCLOUD:
            print_message("Generating word clouds (long processes)")
            generate_word_clouds()
        write_results(divide_results(results))
        send_work_done(glbs.TRAIN_DIR)
        print_message("Done!")
        # clean_backup_files()
    except Exception as e:
        traceback.print_exc()
        send_work_done(
            glbs.TRAIN_DIR, "", error=str(e), traceback=str(traceback.format_exc())
        )


def get_cfg_files(dir):
    items = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            with open(os.path.join(dir, file), "r", encoding="utf8", errors="replace") as f:
                items.append((file.replace(".json", ""), json.load(f)))
    return items


def clean_backup_files():
    glbs = GlobalParameters()
    print_message("removing temp files...")
    folder_path = os.sep.join(glbs.RESULTS_PATH.split(os.sep)[:-1]) + os.sep + "temp_backups"
    shutil.rmtree(folder_path, ignore_errors=True)


if __name__ == "__main__":
    cfg_dir =  sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Mickey\Documents\kerner\textclassificationscript\cfgs"
    if not os.path.exists(cfg_dir):
        cfg_dir = r"C:\Users\user\Documents\test\cfgs"
    main(cfg_dir)

