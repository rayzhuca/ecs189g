from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test import Setting_Train_Test
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- objection initialization section ---------------
    data = Dataset_Loader('mnist', 'handwritten digits')
    data.dataset_source_folder_path = '../../data/stage_3_data/'
    data.dataset_source_file_name = 'MNIST'

    method_obj = Method_CNN_MNIST('CNN MNIST', 'CNN model for MNIST')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/'
    result_obj.result_destination_file_name = 'mnist_prediction_result'

    setting_obj = Setting_Train_Test('train and test', 'runs a training dataset and a testing dataset')

    evaluate_obj = Evaluate_Accuracy('accuracy', 'macro f1, recall, precision, and accuracy')

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