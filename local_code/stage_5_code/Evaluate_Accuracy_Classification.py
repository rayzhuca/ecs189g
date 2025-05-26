# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


class Evaluate_Accuracy(evaluate):
    data = None

    def __init__(self, eName=None, eDescription=None, show_plot=True):
        super().__init__(eName, eDescription)
        self.show_plot = show_plot

    def evaluate(self):
        print('evaluating performance...')

        ret = {}

        epochs = list(range(len(self.data["losses"])))
        if self.show_plot:
            plt.plot(epochs, self.data["losses"], label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            ret["losses"] = self.data["losses"]


        for key in ["train", "test"]:
            print('performance for ', key)

            true_y = self.data[key + '_' + 'true_y']
            pred_y = self.data[key + '_' + 'pred_y']

            accuracy = accuracy_score(true_y, pred_y)
            precision = precision_score(true_y, pred_y, average='macro')
            recall = recall_score(true_y, pred_y, average='macro')
            f1 = f1_score(true_y, pred_y, average='macro')

            ret = ret | {
                key + " accuracy": accuracy, key + " macro precision": precision, key + " macro recall": recall,
                key + " macro f1": f1
            }

        return ret