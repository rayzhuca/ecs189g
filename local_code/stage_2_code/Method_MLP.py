'''
Concrete MethodModule class for a specific learning MethodModule
'''
from torch.utils.data import DataLoader, TensorDataset

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.fc_layer_1 = nn.Linear(784, 200)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(200, 10)
        self.activation_func_2 = nn.Softmax(dim=1)
        self.losses = []

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        # x = x / 255.0
        h1 = self.activation_func_1(self.fc_layer_1(x))
        y_pred = self.activation_func_2(self.fc_layer_2(h1))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        dataloader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=128, shuffle=True)

        for epoch in range(self.max_epoch):

            for id_batch, (X_batch, y_batch) in enumerate(dataloader):

                y_pred = self.forward(torch.FloatTensor(np.array(X_batch)))
                y_true = torch.LongTensor(np.array(y_batch))

                # calculate the training loss
                train_loss = loss_function(y_pred, y_true) + 1e-4 * (torch.norm(self.fc_layer_1.weight, 2) + torch.norm(self.fc_layer_2.weight, 2))
                # print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss))
                if id_batch == dataloader.batch_size - 1:
                    self.losses.append(train_loss.item())

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                if id_batch%20 == 0:
                    accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                    print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')

        print('train size', len(self.data['train']['y']))
        print('test size', len(self.data['test']['y']))

        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        train_pred_y = self.test(self.data['train']['X'])
        test_pred_y = self.test(self.data['test']['X'])

        print(self.data['test']['y'][:100])
        print(test_pred_y[:100])

        return {'train_pred_y': train_pred_y, 'test_pred_y': test_pred_y, 'test_true_y': self.data['test']['y'], 'train_true_y': self.data['train']['y'], 'losses': self.losses}
            