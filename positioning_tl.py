import tensorflow as tf
import numpy as np
from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Lambda, Concatenate, Add
from keras.layers import Multiply, Reshape, Dropout, Conv3D, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
from data_generator import DataGenerator
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
matplotlib.use('Agg')

num_tl_samples = [500, 1000, 5000, 10000, 50000, 100000]

num_antennas = [64]
scenario = "ULA"
transfered = "URA"
data_path = "/home/sdebast/data/mamimo_measurements/"

# tf.logging.set_verbosity(tf.logging.ERROR)


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


# Definition of the NN
def build_nn(num_antenna=64):
    nn_input = Input((num_antenna, num_sub, 2))

    dropout_rate = 0.25
    num_complex_channels = 6

    def k_mean(tensor):
        return K.mean(tensor, axis=2)

    mean_input = Lambda(k_mean)(nn_input)
    print(mean_input.get_shape())

    # complex to polar

    real = Lambda(lambda x: x[:, :, :, 0])(nn_input)
    imag = Lambda(lambda x: x[:, :, :, 1])(nn_input)

    # complex_crop = Lambda(lambda x: x[:, :, 0, :], output_shape=(Nb_Antennas, 2, 1))(complex_input)
    # complex_input = Reshape((Nb_Antennas, 2, 1))(mean_input)

    real_squared = Multiply()([real, real])
    imag_squared = Multiply()([imag, imag])

    real_imag_squared_sum = Add()([real_squared, imag_squared])

    # amplitude
    def k_sqrt(tensor):
        r = K.sqrt(tensor)
        return r

    r = Lambda(k_sqrt)(real_imag_squared_sum)
    r = Reshape((num_antenna, num_sub, 1))(r)
    print(r.get_shape())

    # phase
    def k_atan(tensor):
        import tensorflow as tf
        t = tf.math.atan2(tensor[0], tensor[1])
        return t

    t = Lambda(k_atan)([imag, real])
    t = Reshape((num_antenna, num_sub, 1))(t)
    print(t.get_shape())

    def ifft(x):
        y = tf.complex(x[:, :, :, 0], x[:, :, :, 1])
        ifft = tf.spectral.ifft(y)
        return tf.stack([tf.math.real(ifft), tf.math.imag(ifft)], axis=3)

    polar_input = Concatenate()([r, t])

    time_input = Lambda(ifft)(nn_input)

    total_input = Concatenate()([nn_input, polar_input, time_input])
    # print("total", total_input.get_shape())


    # reduce dimension of time axis

    lay_input = Reshape((num_antenna, num_sub, num_complex_channels, 1))(total_input)

    layD1 = Conv3D(8, (1, 23, num_complex_channels), strides=(1, 5, 1), padding='same')(lay_input)
    layD1 = LeakyReLU(alpha=0.3)(layD1)
    layD1 = Dropout(dropout_rate)(layD1)
    layD2 = Conv3D(8, (1, 23, 1), padding='same')(layD1)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Concatenate()([layD1, layD2])
    layD2 = Conv3D(8, (1, 1, num_complex_channels), padding='same')(layD2)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Conv3D(8, (1, 23, 1), strides=(1, 5, 1), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(layD2)
    layD2 = LeakyReLU(alpha=0.3)(layD2)
    layD2 = Dropout(dropout_rate)(layD2)
    layD3 = Conv3D(8, (1, 23, 1), padding='same')(layD2)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Concatenate()([layD2, layD3])
    layD3 = Conv3D(8, (1, 1, num_complex_channels), padding='same')(layD3)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Conv3D(8, (1, 23, 1), strides=(1, 5, 1), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(layD3)
    layD3 = LeakyReLU(alpha=0.3)(layD3)
    layD3 = Dropout(dropout_rate)(layD3)
    layD4 = Conv3D(8, (1, 23, 1), padding='same')(layD3)
    layD4 = LeakyReLU(alpha=0.3)(layD4)
    layD4 = Concatenate()([layD4, layD3])
    layD4 = Conv3D(8, (1, 1, num_complex_channels), padding='same')(layD4)
    layD4 = LeakyReLU(alpha=0.3)(layD4)
    # layD4 = Conv3D(8, (1, 23, 1), strides=(1, 5, 1), padding='same',
    #                kernel_regularizer=regularizers.l2(0.01))(layD4)
    # layD4 = LeakyReLU(alpha=0.3)(layD4)
    # layD4 = Dropout(dropout_rate)(layD4)

    # conv over antenna layers

    layV1 = Conv3D(8, (8, 1, 1), padding='same')(layD4)
    layV1 = LeakyReLU(alpha=0.3)(layV1)
    layV1 = Dropout(dropout_rate)(layV1)
    layV1 = Concatenate()([layV1, layD4])
    layV2 = Conv3D(8, (8, 1, 1), padding='same',
                   kernel_regularizer=regularizers.l2(0.01))(layV1)
    layV2 = LeakyReLU(alpha=0.3)(layV2)
    layV2 = Dropout(dropout_rate)(layV2)
    layV2 = Concatenate()([layV2, layV1])
    layV3 = Conv3D(8, (8, 1, 1), padding='same')(layV2)
    layV3 = LeakyReLU(alpha=0.3)(layV3)
    layV3 = Dropout(dropout_rate)(layV3)
    layV3 = Concatenate()([layV3, layV2])
    layV4 = Conv3D(8, (8, 1, 1), padding='same')(layV3)
    layV4 = LeakyReLU(alpha=0.3)(layV4)
    layV4 = Dropout(dropout_rate)(layV4)
    layV4 = Concatenate()([layV4, layV3])
    layV5 = Conv3D(8, (8, 1, 1), padding='same')(layV4)
    layV5 = LeakyReLU(alpha=0.3)(layV5)
    layV5 = Dropout(dropout_rate)(layV5)
    # layV3 = Dropout(dropout_rate)(layV3)
    # layV4 = Conv3D(8, (16, 1, 1), padding='valid')(layV3)
    # layV4 = LeakyReLU(alpha=0.3)(layV4)
    # layV4 = Dropout(dropout_rate)(layV4)
    # layV5 = Conv3D(8, (16, 1, 1), padding='valid',
    #                kernel_regularizer=regularizers.l2(0.01))(layV4)
    # layV5 = LeakyReLU(alpha=0.3)(layV5)
    # layV5 = Dropout(dropout_rate)(layV5)
    # layV6 = Conv3D(8, (16, 1, 1), padding='valid')(layV5)
    # layV6 = LeakyReLU(alpha=0.3)(layV6)
    # layV6 = Dropout(dropout_rate)(layV6)

    # conv over complex layers

    # layH1 = Conv3D(12, (1,1,2), strides=(1,1,2), padding='valid', activation='relu')(layV3)
    # layH1 = Dropout(dropout_rate)(layH1)
    # layH2 = Conv3D(12, (1,1,2), padding='same', activation='relu')(layH1)
    # layH2 = Dropout(dropout_rate)(layH2)
    # layH2 = Concatenate()([layH1, layH2])
    # layH3 = Conv3D(12, (1,1,4), padding='same', activation='relu')(layH2)
    # layH3 = Dropout(dropout_rate)(layH3)

    nn_output = Flatten()(layV5)
    nn_output = Dense(64, activation='relu')(nn_output)
    nn_output = Dense(32, activation='relu')(nn_output)
    nn_output = Dense(2, activation='linear')(nn_output)
    nn = Model(inputs=nn_input, outputs=nn_output)
    nn.compile(optimizer='Adam', loss='mse', metrics=[dist])
    nn.summary()
    return nn


num_samples = 252004

# Training size
validation_size = 0.1
test_size = 0.05

# Number of Antennas
# num_antennas = 64
num_sub = 100
nb_epoch = 1000

labels = np.load(data_path + 'labels.npy')


for nb_train_samples in num_tl_samples:
    print("transfer learning with", nb_train_samples, "training smaples")
    bad_samples = np.load("bad_channels_" + scenario + ".npy")
    IDs = []
    for x in range(num_samples):
        if x not in bad_samples:
            IDs.append(x)
    IDs = np.array(IDs)
    np.random.seed(64)   # same random seed to have same dataset each time
    np.random.shuffle(IDs)
    actual_num_samples = IDs.shape[0]
    # nb_train_samples = int(trainings_size*actual_num_samples)
    # nb_train_samples = 1000
    nb_val_samples = int(validation_size*actual_num_samples)
    nb_test_samples = int(test_size*actual_num_samples)
    train_IDs = IDs[:nb_train_samples]
    val_IDs = IDs[nb_train_samples:nb_train_samples+nb_val_samples]
    test_IDs = IDs[-nb_test_samples:]
    for num_antenna in num_antennas:
        print("scenario:", scenario, "number of antennas:", num_antenna)
        nn = load_model('bestmodels/best_model_ifft_URA_' + str(num_antenna) + '.h5', custom_objects={"tf": tf, "dist": dist})
        nn.summary()
        val_generator = DataGenerator(scenario, val_IDs, labels,
                                      num_antennas=num_antenna,
                                      data_path=data_path)
        test_generator = DataGenerator(scenario, test_IDs, labels,
                                       shuffle=False, num_antennas=num_antenna,
                                       data_path=data_path)

        # nb_epoch = 20
        batch_size = 32
        while batch_size > nb_train_samples:
            batch_size = batch_size/2

        val_dist_hist = []
        train_dist_hist = []

        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        mc = ModelCheckpoint('bestmodels/best_model_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.h5', monitor='val_dist', mode='min', verbose=1, save_best_only=True)
        train_generator = DataGenerator(scenario, train_IDs, labels,
                                        batch_size=batch_size, num_antennas=num_antenna,
                                        data_path=data_path)
        train_hist = nn.fit_generator(train_generator, epochs=nb_epoch,
                                      validation_data=val_generator,
                                      callbacks=[es, mc])
        val_dist_hist.extend(train_hist.history['val_dist'])
        train_dist_hist.extend(train_hist.history['dist'])
        np.save('positioning_model_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', nn.get_weights())
        np.save('val_dist_hist_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', val_dist_hist)
        np.save('train_dist_hist_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', train_dist_hist)
        # plot training history
        plt.figure()
        plt.plot(train_dist_hist, label="dist")
        plt.plot(val_dist_hist, label='val_dist')
        plt.title("Train and validation distance error during the training period")
        plt.legend()
        # plt.ylim([0, 1000])
        plt.ylabel("Distance error [mm]")
        plt.xlabel("Number of epochs")
        plt.savefig('train_hist_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + ".png", bbox_inches='tight', pad_inches=0)
        # nn.save('bestmodels/best_model_tl_' + scenario + '_' + str(num_antenna) + '.h5')
#
        # Load best model to evaluate it's performance on the test set
        nn = load_model('bestmodels/best_model_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.h5', custom_objects={"tf": tf, "dist": dist})
        # r_Positions_pred_train = nn.predict_generator(train_generator)
        r_Positions_pred_test = nn.predict_generator(test_generator)
        test_length = r_Positions_pred_test.shape[0]
        # errors_train = true_dist(Positions_train, r_Positions_pred_train)
        errors_test = true_dist(labels[test_IDs[:test_length]], r_Positions_pred_test)
        np.save('pred_test_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', r_Positions_pred_test)
        np.save('label_test_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', labels[test_IDs[:test_length]])
        #
        # Mean_Error_Train = np.mean(np.abs(errors_train))
        Mean_Error_Test = np.mean(np.abs(errors_test))
        # print('{:<40}{:.4f}'.format('Mean error on Train area: ', Mean_Error_Train))
        print("results for the " + scenario + " scenario with " + str(num_antenna) + " antennas. Trained with", nb_train_samples, "samples:")
        print('\033[1m{:<40}{:.4f}\033[0m'.format('Performance P: Mean error on Test area: ', Mean_Error_Test), 'mm')
        result_file = open('results.txt', 'a')
        result_file.write("results for the " + scenario + " scenario with " + str(num_antenna) + " antennas. Trained with " + str(nb_train_samples) + " samples:\n")
        result_file.write('\033[1m{:<40}{:.4f}\033[0m'.format('Performance P: Mean error on Test area: ', Mean_Error_Test) + 'mm\n')

        # errors = true_dist(r_Positions_pred_test, labels[test_IDs])
        plt.figure()
        plt.hist(errors_test, bins=128, range=(0, 500))
        plt.ylabel('Number of occurence')
        plt.xlabel('Distance error [mm]')
        plt.savefig('error_histogram_tl_' + str(nb_train_samples) + ".png", bbox_inches='tight', pad_inches=0)

        # Error Vector over Area in XY
        plt.figure(figsize=(15, 15))

        error_vectors = np.real(r_Positions_pred_test - labels[test_IDs[:test_length]])
        np.save('error_vec_test_tl_' + str(nb_train_samples) + '_' + str(num_antenna) + '.npy', error_vectors)
        afwijking = np.sum(error_vectors, axis=0)
        print("Mean error direction: ", afwijking)
        result_file.write("Mean error direction: " + str(afwijking) + '\n\n')
        result_file.close()
        plt.quiver(np.real(labels[test_IDs][:, 0]), np.real(labels[test_IDs][:, 1]), error_vectors[:, 0], error_vectors[:, 1], errors_test)
        plt.title("Error vectors of the test samples for the " + scenario + " scenario with " + str(num_antenna) + ' antennas')
        plt.xlabel("X position [mm]")
        plt.ylabel("Y position [mm]")
        plt.savefig("error_vector_tl_" + str(nb_train_samples) + '_' + str(num_antenna) + ".png", bbox_inches='tight', pad_inches=0)
