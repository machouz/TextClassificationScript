import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from global_parameters import GlobalParameters


def roc_curve_data(ts_labels, decision):
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, _ = roc_curve(ts_labels, decision)
    roc_auc = auc(fpr, tpr)
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


def plot_roc_curve(data, result_path, method, title=None):
    fpr = data["fpr"]
    tpr = data["tpr"]
    roc_auc = data["roc_auc"]

    plt.figure(figsize=(3.5, 3))
    plt.rc('font', **{'size': 6})
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(method)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(result_path, title) + '.jpg')
    plt.close('all')