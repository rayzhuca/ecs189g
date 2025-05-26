import torch
from torch import nn

from local_code.base_class.method import method
from local_code.stage_5_code.GCNLayer import GCNLayer

class Method_Citeseer(method, nn.Module):
    # max_epoch = 2000
    # learning_rate = 1e-2

    def __init__(self, mName, mDescription, lr=1e-2, max_epoch=2000, weight_decay=1e-6, hidden_layer=64, dropout_rate=0.5):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.lr = lr
        self.max_epoch = max_epoch
        self.weight_decay = weight_decay

        self.gcn1 = GCNLayer(3703, hidden_layer)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn2 = GCNLayer(hidden_layer, 6)

        self.data = None
        self.losses = []

    def forward(self, x, adj):
        x = self.act1(self.gcn1(x, adj))
        x = self.dropout(x)
        return self.gcn2(x, adj)

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        loss_function = nn.CrossEntropyLoss()
        idx_train, idx_val = self.data["train_test_val"]["idx_train"], self.data["train_test_val"]["idx_val"]
        X, y = self.data["graph"]["X"], self.data["graph"]["y"]
        adj = self.data["graph"]["utility"]["A"]

        for epoch in range(self.max_epoch):
            self.train()

            X_batch, y_batch = X[idx_train], y[idx_train]

            y_pred = self.forward(X, adj)
            train_loss = loss_function(y_pred[idx_train], y_batch)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            self.losses.append(train_loss.item())

            self.eval()
            # val_acc = (y_pred[idx_val].argmax(dim=1) == y[idx_val]).float().mean()
            #
            # if epoch % 20 == 0:
            #     print(f'Epoch: {epoch} | Val Acc: {val_acc:.4f} | Train Loss: {train_loss.item():.4f}')

    def test(self, split):
        self.eval()
        idx = self.data["train_test_val"][split]
        with torch.no_grad():
            output = self.forward(self.data["graph"]["X"], self.data["graph"]["utility"]["A"])
        return output[idx].argmax(dim=1), self.data["graph"]["y"][idx]

    def run(self):
        # print('method running...')
        # print('--start training...')

        self.train_model()

        # print('--start testing...')
        train_pred_y, train_true_y = self.test("idx_train")
        test_pred_y, test_true_y = self.test("idx_test")

        return {'train_pred_y': train_pred_y, 'train_true_y': train_true_y,
                'test_pred_y': test_pred_y, 'test_true_y': test_true_y,
                'losses': self.losses}
