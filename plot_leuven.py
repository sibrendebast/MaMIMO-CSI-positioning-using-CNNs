import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, Concatenate, Add, Multiply, Reshape, Dropout, Conv3D, LeakyReLU
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from data_generator import DataGenerator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.logging.set_verbosity(tf.logging.ERROR)

leuven_indices = np.load("indices_leuven.npy")
data_path = "/home/sdebast/data/mamimo_measurements/"
labels = np.load(data_path + 'labels.npy')

num_sub = 100
num_antennas = 64
scenario = "ULA"


# Distance Functions
def dist(y_true, y_pred):
    return tf.reduce_mean((
        tf.sqrt(
            tf.square(tf.abs(y_pred[:, 0] - y_true[:, 0]))
            + tf.square(tf.abs(y_pred[:, 1] - y_true[:, 1]))
        )))


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


bad_samples = np.load("bad_channels_" + scenario + ".npy")
leuven_indices_good = []
for x in leuven_indices:
    if x not in bad_samples:
        leuven_indices_good.append(x)
leuven_labels = labels[leuven_indices_good]
print("number of indices", len(leuven_labels))


# nn = build_nn(num_antennas)
nn = load_model('bestmodels/best_model_tl_100000_64.h5', custom_objects={"tf": tf, "dist": dist})
generator = DataGenerator(scenario, leuven_indices_good, labels,
                          batch_size=16,
                          num_antennas=num_antennas,
                          data_path=data_path,
                          shuffle=False)
pred_pos = nn.predict_generator(generator)
print(pred_pos.shape)
test_length = pred_pos.shape[0]
print("test length", test_length)

errors_test = true_dist(leuven_labels[:test_length], pred_pos)
Mean_Error_Test = np.mean(np.abs(errors_test))

print("MEAN ERROR:", Mean_Error_Test)

fig = plt.figure(figsize=(5, 4.5))
plt.axis('equal')
plt.xlabel("x position [mm]")
plt.ylabel("y position [mm]")
plt.scatter(*zip(*leuven_labels), s=(72./fig.dpi)**2*5, label="Ground truth")
plt.scatter(*zip(*pred_pos), s=(72./fig.dpi)**2*5, label="Prediction")
plt.legend(ncol=2)
plt.savefig("paper_plots/KUleuven.eps", bbox_inches='tight', pad_inches=0)
