import pickle
import os
from tqdm import tqdm

from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import itertools


from preprocess import clean_text


def load_data(dir_name):
    texts = []
    labels = []

    for label_type in ["neg", "pos"]:
        class_dir = os.path.join(dir_name, label_type)
        for fname in tqdm(os.listdir(class_dir), desc=label_type):
            if fname[-4:] == ".txt":
                f = open(os.path.join(class_dir, fname))
                texts.append(clean_text(f.read()))
                f.close()
                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)
    return texts, labels


def create_tokenizer(texts, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters="")
    tokenizer.fit_on_texts(texts)
    print("Found {} unique tokens".format(len(tokenizer.word_index)))

    return tokenizer


def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer(path):
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
