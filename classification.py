# from keras import Sequential
# from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Conv1D, MaxPooling1D
# from keras.preprocessing import sequence
# from keras.preprocessing.text import Tokenizer
import os
import pickle

from keras import Sequential
from keras.layers import (
    Embedding,
    SpatialDropout1D,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from confusion_matrix import accuracy_confusion_matrix
from global_parameters import print_message, GlobalParameters
from model_persistence import save_model
from precision_recall_curve import precision_recall
from roc_curve import roc_curve_data

glbs = GlobalParameters()

methods = {
    "svc": LinearSVC(),
    "rf": RandomForestClassifier(),
    "mlp": MLPClassifier(),
    "lr": LogisticRegression(),
    "mnb": MultinomialNB(),
}


def get_results(ts_labels, prediction, decision):
    # ( "multilabel_confusion_matrix",   multilabel_confusion_matrix(ts_labels, prediction)),
    measures = {
        "accuracy_score": accuracy_score(ts_labels, prediction),
        "f1_score": f1_score(ts_labels, prediction),
        "precision_score": precision_score(ts_labels, prediction),
        "recall_score": recall_score(ts_labels, prediction),
        "roc_auc_score": roc_auc_score(ts_labels, decision),
        "confusion_matrix": confusion_matrix(ts_labels, prediction),
        "roc_curve": roc_curve_data(ts_labels, decision),
        "precision_recall_curve": precision_recall(ts_labels, decision),
        "accuracy_&_confusion_matrix": accuracy_confusion_matrix(ts_labels, prediction),
    }

    return dict([(i, measures[i]) for i in glbs.MEASURE])
    # return accuracy_score(ts_labels, prediction)
    # return stats


def classify(train, tr_labels, test, ts_labels, all_features, num_iteration=1, model_number=0):
    results = {}
    result = []
    le = LabelEncoder()
    le.fit(tr_labels)
    ts_labels = le.transform(ts_labels)
    tr_labels = le.transform(tr_labels)
    print_message("Classifying")

    # if os.path.exists(temp_file_path):
    #  results = load_backup_file(temp_file_path)
    for classifier in glbs.METHODS:
        print_message("running " + str(classifier), num_tabs=1)
        if classifier in results.keys():
            continue

        if classifier == "rnn":
            clf = get_rnn_model(train)
            clf.fit(train, tr_labels, epochs=3, batch_size=64)
        # scores = clf.evaluate(test, ts_labels, verbose=0)
        # acc_score += scores[1]
        else:
            clf = methods[classifier]
            clf.fit(train, tr_labels)
            prediction = clf.predict(test)
            decision = []
            try:
                decision = clf.decision_function(test)
            except:
                decision = clf.predict_proba(test)
                decision = decision[:, 1]

            result = get_results(ts_labels, prediction, decision)

        model_path = os.path.join(glbs.RESULTS_PATH, "Model " + classifier + str(model_number))
        save_model(clf, model_path)
        del clf

        results[classifier] = result
        # save_backup_file(results, temp_file_path)
    # print(results)
    return results


def get_rnn_model(tr_features):
    top_words = tr_features.shape[1]
    model_lstm = Sequential()
    model_lstm.add(
        Embedding(input_dim=top_words, output_dim=256, input_length=top_words)
    )
    model_lstm.add(SpatialDropout1D(0.3))
    model_lstm.add(LSTM(256, dropout=0.3, recurrent_dropout=0.3))
    model_lstm.add(Dense(256, activation="relu"))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Dense(5, activation="softmax"))
    model_lstm.compile(
        loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"]
    )
    return model_lstm


def get_rnn_model2(tr_features):
    top_words = tr_features.shape[1]
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=top_words))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, recurrent_dropout=0.2, dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def gen_file_path():
    name = ""
    for feature in glbs.FEATURES:
        name += feature.upper()
        name += "@"
    name += glbs.NORMALIZATION + "@"
    for method in glbs.METHODS:
        name += method.upper()
        name += "@"

    folder_path = os.sep.join(glbs.RESULTS_PATH.split(os.sep)[:-1]) + os.sep + "temp_backups"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return os.path.join(folder_path, name) + ".pickle"


def save_backup_file(data, path):
    with open(path, "wb+") as file:
        pickle.dump(data, file)


def load_backup_file(path):
    with open(path, "rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    # extract data
    pass
