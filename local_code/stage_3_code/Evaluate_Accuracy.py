'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        epochs = [i / self.data["batches_per_epoch"] for i in range(len(self.data["losses"]))]
        plt.plot(epochs, self.data["losses"], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

        ret = {}

        for key in ["train", "test"]:
            print('performance for ', key)

            true_y = self.data[key + '_' + 'true_y']
            pred_y = self.data[key + '_' + 'pred_y']

            accuracy = accuracy_score(true_y, pred_y)
            precision = precision_score(true_y, pred_y, average='macro')  # or 'micro', 'weighted'
            recall = recall_score(true_y, pred_y, average='macro')
            f1 = f1_score(true_y, pred_y, average='macro')

            ret = ret | {
                key + " accuracy": accuracy, key + " macro precision": precision, key + " macro recall": recall,
                key + " macro f1": f1
            }

        return ret