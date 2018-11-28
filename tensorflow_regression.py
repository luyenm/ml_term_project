from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import time

import numpy as np
START = time.time()

# Creates the model for training
def build_model(type, column):
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(column,)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])
  optimizer = type
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model


def build_Adadelta(column):
    print('AdadeltaOptimizer initialized')
    return build_model(tf.train.AdadeltaOptimizer(), column)


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    elapse = str(time.time() - START) + 's'
    if epoch % 25 == 0: print(elapse)
    print('.', end='')


# Plots data over a continuous graph
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  # plt.ylim([0, 5])
  plt.title('TensorFlow prediction error')
  plt.show()


# Inputs:
#   E: Epochs, or number of iterations to execute
def adadelta_cv(model, train_data, train_labels, test_data, test_labels=None, E=100):
    # Check datatype to be np.ndarray
    if train_data is pd.core.frame.DataFrame:
        train_data = train_data.values
    if train_labels is pd.core.series.Series:
        train_labels = train_labels.values
    if test_data is pd.core.frame.DataFrame:
        test_data = test_data.values
    if test_labels is pd.core.series.Series:
        test_labels = test_labels.values
    # Get the tensorflow model and optimizer to use
    EPOCHS = E
    print('EPOCHS', EPOCHS)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    # Fitting the data
    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.1, verbose=0,
                        callbacks=[early_stop, PrintDot()],
                        workers=8, use_multiprocessing=True)
    if test_labels is None:
        pred = model.predict(test_data, verbose=0)
        output = []
        for item in range(len(pred)):
            output.append(pred[item][0])
        return output
    else:
        [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
        print()
        print("TensorFlow Keras API - Testing set Mean Abs Error: {:7.2f}".format(mae))
        return mae, history