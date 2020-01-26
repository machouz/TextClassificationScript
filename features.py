import os
import random

import numpy as np

from pickle import dump, load
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from scipy.sparse import hstack, vstack
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_selection import SelectFromModel


# region helpful functions
import system_config
from global_parameters import print_message, GlobalParameters
from skipgrams_vectorizer import SkipGramVectorizer
from stylistic_features import get_stylistic_features_vectorizer


glbs = GlobalParameters()


def read_dataset(path):
    data = []
    for category in os.listdir(path):
        with open(
            os.path.join(path, category), "r+", encoding="utf8", errors="ignore"
        ) as read:
            for example in read:
                record = example.rstrip("\n")
                data.append((record, category))
    return data


def is_ngrams(feature):
    return feature.startswith("ngrams")


def extract_ngrams_args(feature):
    parts = feature.split("_")
    count = int(parts[1])
    type = parts[2]
    tfidf = parts[3]
    n = int(parts[4])
    k = int(parts[5])
    if type == "w":
        type = "word"
    elif type == "c":
        type = "char"
    return count, tfidf, type, n, k


def split_train(split, tr_labels, train):
    data = list(zip(train, tr_labels))
    random.shuffle(data)
    test = data[: int(split * len(data))]
    train = data[int(split * len(data)) :]
    train, tr_labels = zip(*train)
    test, ts_labels = zip(*test)
    return train, tr_labels, test, ts_labels


def get_vectorizer(feature):
    count, tfidf, type, n, k = extract_ngrams_args(feature)
    if tfidf == "tfidf":
        tfidf = True
    else:
        tfidf = False

    if k <= 0:
        vectorizer = TfidfVectorizer(
            max_features=count,
            analyzer=type,
            ngram_range=(n, n),
            lowercase=False,
            use_idf=tfidf,
        )

    # TODO: לבדוק איך משלבים את 2 הוקטורים ביחד
    else:
        vectorizer = SkipGramVectorizer(
            max_features=count, analyzer=type, n=n, k=k, lowercase=False
        )

    return vectorizer

    """if k <= 0:
        if tfidf == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=count,
                analyzer=type,
                ngram_range=(n, n),
                lowercase=False,
                use_idf=True,
            )
        elif tfidf == "tf":
            vectorizer = TfidfVectorizer(
                max_features=count,
                analyzer=type,
                ngram_range=(n, n),
                lowercase=False,
                use_idf=False,
            )
        else:
            vectorizer = CountVectorizer(
                max_features=count, analyzer=type, ngram_range=(n, n), lowercase=False
            )

    else:
        vectorizer = SkipGramVectorizer(
            max_features=count, analyzer=type, n=n, k=k, lowercase=False
        )

    return vectorizer, str(n)"""


def add_feature(feature_dict, feature_name, feature):
    feature_dict.append((feature_name, feature))
    return feature_dict


def extract_features(train_dir, test_dir=""):
    print_message("Extracting Features")

    train_data, train_labels, test_data, test_labels = get_data(test_dir, train_dir)
    glbs.LABELS = train_labels + test_labels
    glbs.TRAIN_DATA = train_data

    feature_lst = []
    # add all the N-Grams feature to the list
    for feature in glbs.FEATURES:
        if is_ngrams(feature):
            vectorizer = get_vectorizer(feature)
            feature_lst = add_feature(feature_lst, feature, vectorizer)
    # add all the stylistic features to the list
    for feature in glbs.STYLISTIC_FEATURES:
        vectorizers = get_stylistic_features_vectorizer(feature)
        for i in range(len(vectorizers)):
            feature_lst = add_feature(feature_lst, feature + str(i), vectorizers[i])
    # convert the list to one vectoriazer using FeatureUnion

    all_features = FeatureUnion(feature_lst)
    train_features = all_features.fit_transform(train_data)

    test_features = all_features.transform(test_data)

    return train_features, train_labels, test_features, test_labels, all_features


def get_data(test_dir, train_dir):
    train_data = read_dataset(train_dir)
    random.shuffle(train_data)
    train_data, train_labels = zip(*train_data)
    if glbs.TEST_DIR == "":
        train_data, train_labels, test_data, test_labels = split_train(
            system_config.TEST_SPLIT, train_labels, train_data
        )
    else:
        test_data = read_dataset(test_dir)
        random.shuffle(test_data)
        test_data, test_labels = zip(*test_data)
    return train_data, train_labels, test_data, test_labels
