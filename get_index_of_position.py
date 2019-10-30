import numpy as np


def get_index(x, y):
    labels = np.load('labels.npy')

    coordinate_x_offset = - 1462
    coordinate_y_offset = 1100

    x = x + coordinate_x_offset
    y = y + coordinate_y_offset

    for index, label in enumerate(labels):
        if label[0] == x and label[1] == y:
            return index

    print("not found", x, y)
