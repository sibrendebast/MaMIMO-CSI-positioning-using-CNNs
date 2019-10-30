import matplotlib.pyplot as plt
import numpy as np


styles = [(0, (3, 2, 1, 2, 1, 2)), (0, (3, 2, 1, 2)), ':', '-.', '--', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
num_samples = [500, 1000, 5000, 10000, 50000, 100000]


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


antenna = 64

predictions = []
labels = []
errors = []
for num in num_samples:
    data = np.load('pred_test_tl_' + str(num) + '_' + str(antenna) + '.npy')
    predictions.append(data)
    data = np.load('label_test_tl_' + str(num) + '_' + str(antenna) + '.npy')
    labels.append(data)
    errors.append(true_dist(labels[-1], predictions[-1]))

plt.figure()
plt.title("CDF of the Positioning Error: Transfer Learning")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
for idx, num in enumerate(num_samples):
    # print(colors[idx])
    label = str(num_samples[idx]) + " samples"
    plt.hist(errors[idx], density=True, cumulative=True,
             histtype='step', bins=1000, range=(0, 1000),
             color=plt.cm.gist_rainbow(idx/5), label=label)
plt.grid(which='both')
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend(loc='lower right')
plt.savefig('paper_plots/cdf_tl.eps', bbox_inches='tight', pad_inches=0)

plt.figure()
# plt.title("CDF of the SINR for different path planning algorithms.")
for i, num in enumerate(num_samples):
    # print(i)
    data = np.array(errors[i])
    data = np.sort(data)
    average = sum(data)/len(data)
    # print(labels[i], average)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    # length = len(data)
    # nb_samples = 200
    # step = length / nb_samples
    # idx = [i*step for i in range(nb_samples-1)]
    # data = np.take(data, idx)
    curve_x = [0]
    curve_x.extend(data)
    curve_x.extend([1000])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    # print(len(data))
    label = str(num_samples[i]) + " samples"
    plt.plot(curve_x, curve_y, label=label, linestyle=styles[i])#,
             # color=plt.tab10.gist_rainbow(i/5.0))
    # print(len(sinrs[i]))
    # plt.hist(sinrs[i], density=True, cumulative=True,
    #          label=labels[i], histtype='step', bins=250)
# print("Histograms created")
font_size = 10
plt.title("CDF of the Positioning Error: Transfer Learning")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 400, -0.1, 1.1])
# plt.show()
# print("Saving plots")
plt.savefig('paper_plots/cdf_tl_fix.eps',
            bbox_inches='tight', format='eps')
