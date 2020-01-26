import os

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from inspect import signature


def precision_recall(ts_labels, decision):
    precision, recall, _ = precision_recall_curve(ts_labels, decision)
    return {"precision": precision, "recall": recall}


def plot_precision_recall_curve(prc, result_path, title=None):
    precision = prc["precision"]
    recall = prc["recall"]

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})

    plt.figure(figsize=(3.5, 3))
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.savefig(os.path.join(result_path, title) + '.jpg', bbox_inches='tight')
    plt.close('all')