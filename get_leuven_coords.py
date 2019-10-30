import numpy as np
import matplotlib.pyplot as plt
from get_index_of_position import get_index


def nround(x, base=5):
    return base * round(x/base)


xy = [[(0, 0), (0, 1250), (1250, 1250), (1250, 0), (0, 0)],
      [(1624, 0), (1624, 1250), (2874, 1250), (2874, 0), (1624, 0)],
      [(0, 1645), (0, 2895), (1250, 2895), (1250, 1645), (0, 1645)],
      [(1624, 1645), (1624, 2895), (2874, 2895), (2874, 1645), (1624, 1645)]]

LEUVEN = []
step_size = 5

height = 600

# make L
L = []
x = 10
y = 100 + height
L.append((x, y))
for i in range(100):
    y += step_size
    L.append((x, y))
for i in range(20):
    x += step_size
    L.append((x, y))
for i in range(80):
    y -= step_size
    L.append((x, y))
for i in range(50):
    x += step_size
    L.append((x, y))
for i in range(20):
    y -= step_size
    L.append((x, y))
for i in range(70):
    x -= step_size
    L.append((x, y))

LEUVEN.extend(L)

# make E
E = []
x = 410
y = 100 + height
E.append((x, y))
for i in range(100):
    y += step_size
    E.append((x, y))
for i in range(70):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x -= step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x -= step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(70):
    x -= step_size
    E.append((x, y))
LEUVEN.extend(E)


# make E
U = []
x = 810
y = 300 + height
U.append((x, y))
for i in range(60):
    y += step_size
    U.append((x, y))
for i in range(20):
    x += step_size
    U.append((x, y))
for i in range(60):
    y -= step_size
    U.append((x, y))
radius = 100
center = (x + radius, y)
for i in range(0, 181, 5):
    x = center[0] - radius*np.sin((i + 90)/180*3.1415)
    y = center[1] + radius*np.cos((i + 90)/180*3.1415)
    U.append((x, y))
for i in range(60):
    y += step_size
    U.append((x, y))
for i in range(20):
    x += step_size
    U.append((x, y))
for i in range(60):
    y -= step_size
    U.append((x, y))
# center = (x + 50, y)
radius += 100
for i in range(0, 181, 3):
    x = center[0] - radius*np.sin((i + 90)/180*3.1415)
    y = center[1] + radius*np.cos((i + 90)/180*3.1415)
    U.append((x, y))
U_rounded = []
for point in U:
    LEUVEN.extend([(nround(point[0]), nround(point[1]))])

# make V
V = []
x = 1624 + 10
y = 600 + height
V.append((x, y))
for i in range(23):
    x += step_size
    V.append((x, y))
for i in range(20):
    x += step_size
    y -= 3*step_size
    V.append((x, y))
for i in range(20):
    x += step_size
    y += 3*step_size
    V.append((x, y))
for i in range(23):
    x += step_size
    V.append((x, y))
for i in range(33):
    x -= step_size
    y -= 3*step_size
    V.append((x, y))
for i in range(20):
    x -= step_size
    V.append((x, y))
for i in range(33):
    x -= step_size
    y += 3*step_size
    V.append((x, y))
LEUVEN.extend(V)


# make E
E = []
x = 1624 + 10 + 450
y = 100 + height
E.append((x, y))
for i in range(100):
    y += step_size
    E.append((x, y))
for i in range(70):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x -= step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x -= step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(50):
    x += step_size
    E.append((x, y))
for i in range(20):
    y -= step_size
    E.append((x, y))
for i in range(70):
    x -= step_size
    E.append((x, y))
LEUVEN.extend(E)


N = []
x = 1624 + 10 + 860
y = 100 + height
for i in range(100):
    y += step_size
    N.append((x, y))
for i in range(20):
    x += step_size
    N.append((x, y))
for i in range(25):
    x += step_size
    y -= 2*step_size
    N.append((x, y))
for i in range(50):
    y += step_size
    N.append((x, y))
for i in range(20):
    x += step_size
    N.append((x, y))
for i in range(100):
    y -= step_size
    N.append((x, y))
for i in range(20):
    x -= step_size
    N.append((x, y))
for i in range(25):
    x -= step_size
    y += 2*step_size
    N.append((x, y))
for i in range(50):
    y -= step_size
    N.append((x, y))
for i in range(20):
    x -= step_size
    N.append((x, y))

LEUVEN.extend(N)


K = []
x = 50
y = 1680
for i in range(220):
    y += step_size
    K.append((x, y))
for i in range(60):
    x += step_size
    K.append((x, y))
for i in range(80):
    y -= step_size
    K.append((x, y))
for i in range(80):
    x += step_size
    y += step_size
    K.append((x, y))
for i in range(70):
    x += step_size
    K.append((x, y))
for i in range(80):
    x -= step_size
    y -= step_size
    K.append((x, y))
for i in range(70):
    x += step_size
    y -= 2*step_size
    K.append((x, y))
for i in range(60):
    x -= step_size
    K.append((x, y))
for i in range(55):
    x -= step_size
    y += 2*step_size
    K.append((x, y))
for i in range(25):
    x -= step_size
    y -= step_size
    K.append((x, y))
for i in range(85):
    y -= step_size
    K.append((x, y))
for i in range(60):
    x -= step_size
    K.append((x, y))

LEUVEN.extend(K)

# make E
U = []
x = 1624 + 200
y = 1680 + 500
U.append((x, y))
for i in range(120):
    y += step_size
    U.append((x, y))
for i in range(50):
    x += step_size
    U.append((x, y))
for i in range(120):
    y -= step_size
    U.append((x, y))
radius = 250
center = (x + radius, y)
for i in range(0, 181, 2):
    x = center[0] - radius*np.sin((i + 90)/180*3.1415)
    y = center[1] + radius*np.cos((i + 90)/180*3.1415)
    U.append((x, y))
for i in range(120):
    y += step_size
    U.append((x, y))
for i in range(50):
    x += step_size
    U.append((x, y))
for i in range(120):
    y -= step_size
    U.append((x, y))
# center = (x + 50, y)
radius += 250
for i in range(0, 181, 1):
    x = center[0] - radius*np.sin((i + 90)/180*3.1415)
    y = center[1] + radius*np.cos((i + 90)/180*3.1415)
    U.append((x, y))
U_rounded = []
for point in U:
    LEUVEN.extend([(nround(point[0])+4, nround(point[1]))])

# plt.figure()
# for pos in xy:
#     print(pos)
#     plt.plot(*zip(*pos))
# plt.scatter(*zip(*LEUVEN))
# plt.axis('equal')
# plt.show()


LEUVEN_indices = []
for point in LEUVEN:
    LEUVEN_indices.append(get_index(point[0], point[1]))

print("number of indices found:", len(LEUVEN_indices))

np.save("indices_leuven", np.array(LEUVEN_indices))

labels = np.load('labels.npy')
plt.figure()
plt.scatter(*zip(*labels[LEUVEN_indices]))
plt.axis('equal')
plt.show()
