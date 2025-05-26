import os

import numpy as np
import torch
from ray.tune import CLIReporter
import matplotlib.pyplot as plt
from ray.util.client import ray

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Evaluate_Accuracy_Classification import Evaluate_Accuracy
from local_code.stage_5_code.Method_Pubmed import Method_Pubmed
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test import Setting_Train_Test
from ray import tune


# ---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    np.random.RandomState(2)
    torch.manual_seed(2)

    # ---- objection initialization section ---------------
    data = Dataset_Loader(None, 'pubmed', 'pubmed dataset')
    data.dataset_source_folder_path = os.path.abspath("../../data/stage_5_data/pubmed")

    def train(config):
        np.random.RandomState(2)
        torch.manual_seed(2)

        method_obj = Method_Pubmed('pubmed gcn', '', lr=config['lr'], max_epoch=config['max_epoch'], weight_decay=config['weight_decay'],
                                     hidden_layer=config['hidden_layer'], dropout_rate=config['dropout_rate'])

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_5_result/'
        result_obj.result_destination_file_name = 'pubmed_result'

        setting_obj = Setting_Train_Test('train and test', 'runs a training dataset and a testing dataset')

        evaluate_obj = Evaluate_Accuracy('accuracy', 'macro f1, recall, precision, and accuracy', show_plot=False)

        setting_obj.prepare(data, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        evaluation_data = setting_obj.load_run_save_evaluate()

        tune.report({"mean_accuracy": evaluation_data["test accuracy"], "evaluation_data": evaluation_data})


    # ---- running section ---------------------------------
    print('************ Start ************')

    analysis = tune.run(
        train, config={"lr": tune.grid_search([0.01]),
                    "max_epoch": tune.grid_search([300]),
                    "weight_decay": tune.grid_search([1e-4]),
                    "hidden_layer": tune.grid_search([128]),
                   "dropout_rate": tune.grid_search([0.6]),
        }, progress_reporter=CLIReporter(metric_columns=["mean_accuracy"]),
        metric="mean_accuracy",
        mode="max")

    best_trial = analysis.get_best_trial(metric="mean_accuracy", mode="max", scope="all")
    best_eval_data = best_trial.last_result.get("evaluation_data", {})

    print('************ Overall Performance ************')
    print('Classification Performance: ')

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    epochs = list(range(len(best_eval_data["losses"])))
    plt.plot(epochs, best_eval_data["losses"], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    for k, v in best_eval_data.items():
        if k != "losses":
            print(f"{k}: {v}")

    print('************ Finish ************')
    # ------------------------------------------------------