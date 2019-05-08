#!/usr/bin/env python

from data import X_train, X_test, y_train, y_test
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adam
from evaluation import evaluate_model
from keras.callbacks import EarlyStopping

# create the models
model = Sequential()
model.add(Dense(10, input_shape=(X_train.shape[1],)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(),
                  metrics=["accuracy"])
model.summary()

#callbacks = [EarlyStopping(monitor='val_loss', patience=1)]

history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=10,
                        shuffle=True,
                        #callbacks=callbacks,
                        #validation_split=0.1,
                        verbose=1
                        )

#print (history.history['val_loss'])

# evaluation
evaluate_model(model, X_test, y_test)
