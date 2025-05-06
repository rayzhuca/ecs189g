'''
Concrete IO class for a specific dataset
'''

from matplotlib import pyplot as plt
import pickle
import numpy as np
from local_code.base_class.dataset import dataset
import torch

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print(f'loading dataset {self.dataset_name} ...')
        with open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb') as f:
            data = pickle.load(f)

        ret = {}

        for key in ["train", "test"]:
            X, y = [], []
            for instance in data[key]:
                X.append(instance["image"])
                y.append(instance["label"])

            X, y = torch.tensor(np.array(X), dtype=torch.float32), torch.LongTensor(np.array(y))
            if X.dim() == 3:
                X = X.unsqueeze(-1)

            X = X.permute(0, 3, 1, 2)

            if self.dataset_name == "orl":
                X = X[:, 0:1, :, :]
                y -= 1

            X.contiguous()

            ret[key] = {"X": X, "y": y}

        return ret