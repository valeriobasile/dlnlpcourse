import spacy
import os
import logging as log
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def read_file(data_file):
    try:
        f = open(data_file, encoding="latin1")
    except:
        log.error("cannot open file: {0}".format(data_file))
        return None, None

    sentences = []
    labels = []
    for line in f:
        try:
            sentence = " ".join(line.strip().split("\t")[:-1])
            label = line.strip().split("\t")[-1]
            sentences.append(sentence)
            labels.append(label)
        except:
            log.warning("error reading line {0} of file {1}".format(len(sentences), data_file)),
    log.info("read {0} instances".format(len(sentences)))
    return sentences, labels

def read_preprocessed_file(data_file):
    try:
        f = open(data_file)
    except:
        log.error("cannot open file: {0}".format(data_file))
        return None, None

    sentences = []
    labels = []
    tokens = []
    for line in f:
        if line.startswith("<label>"):
            label = line.strip().replace("<label>", "")
            labels.append(label)
        elif line != "\n":
            tokens.append(line.strip())
        else:
            sentences.append(tokens)
            tokens = []

    log.info("read {0} instances ({1} labels)".format(len(sentences), len(labels)))
    return sentences, labels

log.info("preprocessing the training data")
sentences, labels = read_file("data/train.tsv")

log.info("preprocessing the sentences")
if os.path.isfile("data/train_preprocessed.tsv"):
    sentences_preprocessed, labels = read_preprocessed_file("data/train_preprocessed.tsv")
else:
    nlp = spacy.load("it", disable=["ner", "pos", "parser"])
    sentences_preprocessed = []
    for sentence in sentences:
        doc = nlp(sentence)
        sentences_preprocessed.append([token.lemma_ for token in doc])
    with open("data/train_preprocessed.tsv", "w") as fo:
        for sentence_preprocessed, label in zip(sentences_preprocessed, labels):
            fo.write("<label>{0}\n".format(label))
            for token in sentence_preprocessed:
                fo.write("{0}\n".format(token))
            fo.write("\n")

# transform the sentences into vectors
log.info("vectorization")
tokenizer = Tokenizer(filters='', lower=True, split=' ')
tokenizer.fit_on_texts(sentences_preprocessed)
word_index = tokenizer.word_index
X_data = tokenizer.texts_to_matrix(sentences_preprocessed)
X_data = pad_sequences(X_data, 280, padding='post', truncating='post')

# encode the labels
labels = np.array(labels)
encoder = LabelEncoder()
encoder.fit(labels)
y_data = encoder.transform(labels)
y_data = to_categorical(y_data)

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)
