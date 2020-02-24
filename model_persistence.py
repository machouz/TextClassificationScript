import pickle

def save_model(clf, path):
    with open(path, 'wb') as f:
        pickle.dump(clf, f)


def load_model(path):
    with open(path, 'wb') as f:
        return pickle.load(f)
