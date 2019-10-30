import numpy as np
import matplotlib.pyplot as plt


scenarios = ["URA", "ULA", "distributed"]
styles = [':', '-.', '--', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

num_antennas = [8, 16, 32, 64]


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


predictions = []
labels = []
errors = []
for scenario in scenarios:
    for antenna in num_antennas:
        data = np.load('test_eva/pred_test_ifft_' + scenario + '_' + str(antenna) + '.npy')
        predictions.append(data)
        data = np.load('test_eva/label_test_ifft_' + scenario + '_' + str(antenna) + '.npy')
        labels.append(data)
        errors.append(true_dist(labels[-1], predictions[-1]))

plt.figure()
plt.title("CDF of the Positioning Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
i = 0
for idx, scenario in enumerate(scenarios):
    for jdx, antenna in enumerate(num_antennas):
        # print(colors[idx])
        plt.hist(errors[i], density=True, cumulative=True, linestyle=styles[jdx],
                 histtype='step', bins=500, range=(0, 500), color=colors[idx],
                 label=scenarios[idx])
        i += 1
plt.grid()
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend(loc='lower right')
plt.savefig('paper_plots/cdf_antenna.eps', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.title("CDF of the Positioning Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
for idx, scenario in enumerate(scenarios):
    plt.hist(errors[3+idx*4], density=True, cumulative=True, linestyle=styles[idx],
             histtype='step', bins=300, range=(0, 300), color=colors[idx],
             label=scenarios[idx])
plt.grid()
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend(loc='lower right')
plt.savefig('paper_plots/cdf_scenario.eps', bbox_inches='tight', pad_inches=0)


plt.figure()
# plt.title("CDF of the SINR for different path planning algorithms.")
for i, scenario in enumerate(scenarios):
    # print(i)
    data = np.array(errors[3+i*4])
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
    curve_x.extend([500])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    # print(len(data))
    plt.plot(curve_x, curve_y, label=scenario, linestyle=styles[i])
    # print(len(sinrs[i]))
    # plt.hist(sinrs[i], density=True, cumulative=True,
    #          label=labels[i], histtype='step', bins=250)
# print("Histograms created")
font_size = 10
plt.title("CDF of the Positioning Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 300, -0.1, 1.1])
# plt.show()
# print("Saving plots")
plt.savefig('paper_plots/cdf_scenario_fix.eps',
            bbox_inches='tight', format='eps')
# print(".png done")
