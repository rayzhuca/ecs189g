'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test(setting):

    def load_run_save_evaluate(self):
        data = self.dataset.load()

        self.method.data = data
        learned_result = self.method.run()

        # self.result.data = learned_result
        # self.result.save()

        self.evaluate.data = learned_result

        evaluation = self.evaluate.evaluate()

        return evaluation