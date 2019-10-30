import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, scenario, list_IDs, labels, batch_size=32, num_antennas=64,
                 num_subc=100, n_channels=2, shuffle=True, data_path="/volume1/scratch/sdebast/mamimo_measurements/"):
        # 'Initialization'
        self.dim = (num_antennas, num_subc)
        if num_antennas == 64:
            self.antennas = [x for x in range(64)]
        elif num_antennas == 32:
            if scenario == "distributed":
                self.antennas = [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21,
                                 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44,
                                 45, 50, 51, 52, 53, 58, 59, 60, 61]
            elif scenario == "URA":
                self.antennas = [10, 11, 12, 13, 17, 18, 19, 20, 21, 22,
                                 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38,
                                 41, 42, 43, 44, 45, 46, 50, 51, 52, 53]
            elif scenario == "ULA":
                self.antennas = [x + 16 for x in range(32)]
        elif num_antennas == 16:
            if scenario == "distributed":
                self.antennas = [3, 4, 11, 12, 19, 20, 27, 28,
                                 35, 36, 43, 44, 51, 52, 59, 60]
            elif scenario == "URA":
                self.antennas = [18, 19, 20, 21, 26, 27, 28, 29,
                                 34, 35, 36, 37, 42, 43, 44, 45]
            elif scenario == "ULA":
                self.antennas = [x + 24 for x in range(16)]
        elif num_antennas == 8:
            if scenario == "distributed":
                self.antennas = [3 + 8*x for x in range(8)]
            elif scenario == "URA":
                self.antennas = [26, 27, 28, 29,
                                 34, 35, 36, 37]
            elif scenario == "ULA":
                self.antennas = [x + 28 for x in range(8)]
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data_path = data_path + "channel_measurements_" + scenario + "/"
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            sample = np.load(self.data_path + "channel_measurement_" + str(ID).zfill(6) + '.npy')
            # print(X[i, :, :, 0].shape)
            # print(sample.real.shape)
            X[i, :, :, 0] = sample.real[self.antennas, :]
            X[i, :, :, 1] = sample.imag[self.antennas, :]

            # Store class
            y[i] = self.labels[ID, :]

        return X, y
