import pickle

import torch

from local_code.base_class.result import result

class Model_Saver(result):
    state_dict = None
    vocab = None
    embedding = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving model...')
        torch.save(self.state_dict, self.result_destination_folder_path + self.result_destination_file_name)
        with open(self.result_destination_folder_path + 'vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
        with open(self.result_destination_folder_path + 'embedding.pkl', 'wb') as f:
            pickle.dump(self.embedding, f)