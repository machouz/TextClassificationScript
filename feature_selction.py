import os
import pickle
from array import array
from keras import Sequential
from scipy.sparse import hstack, vstack
from keras.layers import (
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
)
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tensorflow import one_hot

from confusion_matrix import accuracy_confusion_matrix
from features import get_data
from global_parameters import print_message, GlobalParameters
from help_functions import write_result
from precision_recall_curve import precision_recall
from roc_curve import roc_curve_data
from sklearn.feature_selection import (
    chi2,
    mutual_info_classif,
    f_classif,
    RFECV,
    mutual_info_regression,
    SelectKBest,
    SelectFromModel,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from new_xlsx_file import write_info_gain, write_sfm

glbs = GlobalParameters()

methods = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(),
    "lr": LogisticRegression(),
    "mnb": MultinomialNB(),
}

selection_type = {
    "chi2": chi2,
    "mir": mutual_info_regression,
    "mic": mutual_info_classif,
    "fc": f_classif,
    "rfecv": RFECV,
    "sfm": SelectFromModel,
}


def get_selection_list(selection, features, labels):
    return selection_type[selection](features, labels)


def select_k_best(selection, k):
    return SelectKBest(selection_type[selection], k=k)


def get_recursive(features, labels):
    ranking = {}
    for key, method in methods.items():
        recursive = RFECV(
            method, step=1, cv=[(range(134), range(134, 200))], scoring="accuracy"
        )
        train_features = recursive.fit(features, labels)
        ranking[key] = train_features.ranking_
    return ranking


def select_rfecv_sfm(selection, features, labels):
    if selection[0] == "rfecv":
        for key, method in methods.items():
            recursive = RFECV(
                method, step=1, cv=[(range(134), range(134, 200))], scoring="accuracy"
            )
            recursive.fit(features, labels)
            # Plot number of features VS. cross-validation scores
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("accuracy score" + key + " (nb of correct classifications)")
            plt.plot(range(1, len(recursive.grid_scores_) + 1), recursive.grid_scores_)
            plt.savefig(os.path.join(glbs.RESULTS_PATH, key) + ".jpg", bbox_inches="tight")
    if selection[0] == "sfm":
        score = {}
        for key, method in methods.items():
            sfm = SelectFromModel(method, max_features=int(selection[1]))
            train_new = sfm.fit_transform(features[0], labels[0])
            test_new = sfm.transform(features[1])
            clf = method
            clf.fit(train_new, labels[0])
            pred = clf.predict(test_new)
            acc = accuracy_score(labels[1], pred)
            score[key] = acc
        write_sfm(score)


def get_selected_features(selection, train, tr_labels, test, ts_labels, all_features):
    le = LabelEncoder()
    le.fit(tr_labels)
    ts_labels = le.transform(ts_labels)
    tr_labels = le.transform(tr_labels)
    if selection[1] == "0":
        selection_list = get_selection_list(selection[0], train, tr_labels)
        ziped = []
        try:
            ziped = zip(
                all_features.get_feature_names(), selection_list[0], selection_list[1]
            )
        except:
            ziped = zip(all_features.get_feature_names(), selection_list)
        write_info_gain(ziped, str(selection[0]))
        return train, test
    if selection[0] in selection_type.keys():
        if selection[0] == "rfecv":
            features = vstack((train, test))
            select_rfecv_sfm(selection, features, glbs.LABELS)
        elif selection[0] == "sfm":
            select_rfecv_sfm(selection, (train, test), (tr_labels, ts_labels))
        else:
            select = select_k_best(selection[0], int(selection[1]))
            return select.fit_transform(train, tr_labels), select.transform(test)

    # write_info_gain(zip(feat_labels, recursive.ranking_), "rfevc " + key)
    # feat_labels = features.get_feature_names()


if __name__ == "__main__":
    pass
