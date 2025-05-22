from torch.utils.data import DataLoader, TensorDataset, Dataset
from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy

device = "mps"

class Method_CNN_Base(method, nn.Module):
    max_epoch = 10
    learning_rate = 1e-3
    l1_lambda_reg = 0
    l2_lambda_reg = 0

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.optimizer = None

        self.losses = []

    def forward(self, x):
        raise NotImplementedError

    def train_model(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        dataloader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
        self.batches_per_epoch = len(dataloader)

        for epoch in range(self.max_epoch):
            train_loss, y_pred, y_true = None, None, None

            for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # print(X_batch.shape, y_batch.shape)

                y_pred = self.forward(X_batch)
                y_true = y_batch

                l2_reg = self.l2_lambda_reg * (sum(param.norm(2) for param in self.parameters()) if self.l2_lambda_reg else 0)
                l1_reg = self.l1_lambda_reg * (
                    sum(param.abs().sum() for param in self.parameters()) if self.l1_lambda_reg else 0)
                train_loss = loss_function(y_pred, y_true) + l1_reg + l2_reg

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                self.losses.append(train_loss.item())
            accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        self.eval()
        dataloader = DataLoader(TensorDataset(X), batch_size=64, shuffle=False)
        y_pred = []
        with torch.no_grad():
            for id_batch, (X_batch,) in enumerate(dataloader):
                y_pred_batch = self.forward(X_batch.to(device))
                y_pred.append(y_pred_batch.cpu())

        y_pred = torch.cat(y_pred, dim=0)
        self.train()
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')

        print('train size', len(self.data['train']['y']))
        print('test size', len(self.data['test']['y']))

        self.train_model(self.data['train']['X'], self.data['train']['y'])

        print('--start testing...')
        train_pred_y = self.test(self.data['train']['X'])
        test_pred_y = self.test(self.data['test']['X'])

        return {'train_pred_y': train_pred_y, 'test_pred_y': test_pred_y, 'test_true_y': self.data['test']['y'],
                'train_true_y': self.data['train']['y'], 'losses': self.losses, 'batches_per_epoch': self.batches_per_epoch}
