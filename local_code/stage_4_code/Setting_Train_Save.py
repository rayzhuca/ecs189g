'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Save(setting):

    def load_run_save_evaluate(self):
        # load dataset
        data = self.dataset.load()

        # run MethodModule
        self.method.data = data
        res = self.method.run()

        # save model and vocab
        self.result.state_dict = res["state_dict"]
        self.result.vocab = res["vocab"]
        self.result.embedding = res["embedding"]
        self.result.save()

        # evaluate
        self.evaluate.data = res
        return self.evaluate.evaluate()