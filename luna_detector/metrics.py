import pandas as pd
from matplotlib import pyplot as plt
from os.path import join as opjoin
import numpy as np
from path import luna_path

PLOT_SAVE_DIR = 'plots'


class Metrics:
    def __init__(self,
                 roc_auc,
                 min_dist01,
                 juden_index,
                 sensitivity,
                 specificity,
                 accuracy,
                 precision_positive,
                 precision_negative,
                 positive,
                 negative):
        self.roc_auc = roc_auc
        self.min_dist01 = min_dist01
        self.juden_index = juden_index
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.accuracy = accuracy
        self.precision_positive = precision_positive
        self.precision_negative = precision_negative
        self.positive = positive
        self.negative = negative

    @staticmethod
    def save(metrics, filename):
        df = pd.DataFrame({'roc_auc': [m.roc_auc for m in metrics],
                           'minimal_distance_to_01': [m.min_dist01 for m in metrics],
                           'juden_index': [m.juden_index for m in metrics],
                           'sensitivity': [m.sensitivity for m in metrics],
                           'specificity': [m.specificity for m in metrics],
                           'accuracy': [m.accuracy for m in metrics],
                           'precision_positive': [m.precision_positive for m in metrics],
                           'precision_negative': [m.precision_negative for m in metrics],
                           'positive': [m.positive for m in metrics],
                           'negative': [m.negative for m in metrics],
                           })
        df.to_csv(filename, index=False)


fpr = lambda x: (1 - x[1] / x[3])
tpr = lambda x: x[0] / x[2]
dist01 = lambda x: math.sqrt((fpr(x) ** 2) + ((1 - tpr(x)) ** 2))


def dist_mid(x):
    return (tpr(x) - fpr(x)) / math.sqrt(2)


import math


class MetricsCalculator:
    def __init__(self, roc_result, dice_threshold=0):
        roc_result = list(roc_result.values())
        self.dice_threshold = dice_threshold
        roc_result.reverse()
        self.roc_result = roc_result

    def roc_auc(self):
        roc_auc = 0
        for i in range(len(self.roc_result) - 1):
            roc_auc += (fpr(self.roc_result[i + 1]) - fpr(self.roc_result[i])) * (
                    tpr(self.roc_result[i + 1]) + tpr(self.roc_result[i])) / 2
        return roc_auc

    def draw_roc(self, filename):
        plt.ylim(-0.1, 1.2)
        plt.xlim(-0.1, 1.2)
        plt.ylabel('tpr')
        plt.xlabel('fpr')
        plt.grid()
        mind_idx = np.argmin([dist01(x) for x in self.roc_result])
        mind_x = fpr(self.roc_result[mind_idx])
        mind_y = tpr(self.roc_result[mind_idx])
        print(mind_x, mind_y)
        plt.plot([mind_x, 0], [mind_y, 1], color='green', linestyle='dashed', marker='o')

        juden_idx = np.argmax([dist_mid(x) for x in self.roc_result])
        juden_x = fpr(self.roc_result[juden_idx])
        juden_y = tpr(self.roc_result[juden_idx])
        juden_coord = (juden_x + juden_y) / 2
        print(juden_x, juden_y)
        plt.plot([juden_x, juden_coord], [juden_y, juden_coord], color='purple', linestyle='dashed', marker='o')

        plt.plot([0, 1], [0, 1], color='pink', linestyle='dashed', marker='o')
        plt.plot([0, 1], [1, 1], color='orange', linestyle='dashed')
        plt.plot([1, 1], [0, 1], color='orange', linestyle='dashed')
        print([fpr(x) for x in self.roc_result])
        print([x[0] / x[2] for x in self.roc_result])
        plt.plot([fpr(x) for x in self.roc_result], [x[0] / x[2] for x in self.roc_result], linewidth=3)
        ax = plt.axes()

        ax.arrow(0, 0, 0, 1.1, head_width=0.03, head_length=0.04, fc='k', ec='k', color='blue')
        ax.arrow(0, 0, 1.1, 0, head_width=0.03, head_length=0.04, fc='k', ec='k', color='blue')

        plt.savefig(opjoin(luna_path, filename))
        plt.show()

    def calculate(self, tprw=1):
        res = self.roc_result.copy()
        res.sort(key=lambda x: tprw * tpr(x) - fpr(x))
        optimal = res[-1]
        print(optimal)
        m = Metrics(self.roc_auc(),
                    np.min([dist01(x) for x in self.roc_result]),
                    np.max([dist_mid(x) for x in self.roc_result]),
                    tpr(optimal),
                    1 - fpr(optimal),
                    (optimal[0] + optimal[1]) / (optimal[2] + optimal[3]),
                    optimal[0] / (optimal[0] + optimal[3] - optimal[1]),
                    optimal[1] / (optimal[1] + optimal[2] - optimal[0]),
                    optimal[2],
                    optimal[3])
        return m


class FROCMetricsCalculator:
    def __init__(self, froc_result):
        froc_result = list(froc_result.values())
        froc_result.reverse()
        self.roc_result = froc_result

    def roc_auc(self):
        roc_auc = 0
        for i in range(len(self.roc_result) - 1):
            roc_auc += (fpr(self.roc_result[i + 1]) - fpr(self.roc_result[i])) * (
                    tpr(self.roc_result[i + 1]) + tpr(self.roc_result[i])) / 2
        return roc_auc

    def draw_roc(self, filename):
        points = [[res[-1] / res[2], res[0] / res[2]] for res in self.roc_result]
        max_x = 10
        plt.ylim(-0.1, 1.2)
        plt.xlim(-0.1, max_x)
        plt.ylabel('tpr')
        plt.xlabel('average fp / crop')
        plt.grid()

        plt.plot(points[:, 0], points[:, 1], linewidth=3)
        ax = plt.axes()

        ax.arrow(0, 0, 0, 1.1, head_width=max_x / 30, head_length=0.04, fc='k', ec='k', color='blue')
        ax.arrow(0, 0, max_x, 0, head_width=0.03, head_length=max_x / 30, fc='k', ec='k', color='blue')

        plt.savefig(opjoin(luna_path, PLOT_SAVE_DIR, filename))
        plt.show()


    def save(self, filename):
        np.save(opjoin(luna_path, 'roc_results_npy', filename), self.roc_result)
