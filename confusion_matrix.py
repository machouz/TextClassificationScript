import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# import some data to play with
from global_parameters import GlobalParameters


def accuracy_confusion_matrix(ts_labels, prediction):
    return {"accuracy": accuracy_score(ts_labels, prediction), "matrix": confusion_matrix(ts_labels, prediction)}


def plot_confusion_matrix(cm, result_path, normalize=True, title=None, accuracy=None, cmap=plt.cm.Blues, color='black'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    glbs = GlobalParameters()
    labels = [label.split('.')[0] for label in list(set(glbs.LABELS))]

    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           # title=title,
           ylabel='True',
           xlabel='Predicted')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if accuracy:
        accuracy = float('{0:.4g}'.format(accuracy*100))
        plt.title('Accuracy Score: ' + str(accuracy) + '\nConfusion Matrix:')
        plt.rcParams.update({"text.color": color})

    plt.savefig(os.path.join(result_path, title) + '.jpg', bbox_inches='tight')


    plt.close('all')



if __name__ == "__main__":
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    #plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')