import random

import numpy as np
import torch

from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from local_code.stage_5_code.Evaluate_Accuracy_Classification import Evaluate_Accuracy
from local_code.stage_5_code.Method_Cora import Method_Cora
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test import Setting_Train_Test

import os

# ---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    random.seed(2)
    os.environ['PYTHONHASHSEED'] = str(2)

    # ---- objection initialization section ---------------
    data = Dataset_Loader(None, 'cora', 'pubmed dataset')
    data.dataset_source_folder_path = "../../data/stage_5_data/cora"

    method_obj = Method_Cora('cora gcn', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/'
    result_obj.result_destination_file_name = 'cora_result'

    setting_obj = Setting_Train_Test('train and test', 'runs a training dataset and a testing dataset')

    evaluate_obj = Evaluate_Accuracy('accuracy', 'macro f1, recall, precision, and accuracy')

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    evaluation_data = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print('Classification Performance: ')

    for k, v in evaluation_data.items():
        print(k + ': ' + str(v))

    print('************ Finish ************')
    # ------------------------------------------------------