#!/usr/bin/env python

from data import X_train, X_test, y_train, y_test, word_index
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Bidirectional, LSTM
from keras.optimizers import Adam
from evaluation import evaluate_model

# create the models
model = Sequential()
model.add(Embedding(len(word_index)+1, 128, input_shape=(280,)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=4,
                        shuffle=True,
                        verbose=1
                        )

# evaluation
evaluate_model(model, X_test, y_test)
