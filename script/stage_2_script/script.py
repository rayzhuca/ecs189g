from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
from collections import namedtuple

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    train_data = Dataset_Loader('train', 'training data')
    train_data.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data.dataset_source_file_name = 'train.csv'

    test_data = Dataset_Loader('test', 'testing data')
    test_data.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data.dataset_source_file_name = 'test.csv'

    data = namedtuple('data', ['train', 'test', 'dataset_name'])(train_data, test_data, 'train and test data')

    method_obj = Method_MLP('multi-layer perceptron', 'basic MLP model')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test('train and test', 'runs a training dataset and a testing dataset')

    evaluate_obj = Evaluate_Accuracy('accuracy', 'macro f1, recall, precision, and accuracy')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    evaluation_data = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Performance: ')
    for k, v in evaluation_data.items():
        print(k + ': ' + str(v))
    print('************ Finish ************')
    # ------------------------------------------------------


