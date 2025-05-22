import torch
from torch import nn

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy

from tqdm import tqdm

device = "cpu"

class Method_Generation(method, nn.Module):
    max_epoch = 100
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.embedding = None
        self.rnn = None
        self.fc1 = None
        self.act1 = None
        self.fc2 = None
        self.dropout = None
        self.losses = []

    def build(self):
        self.embedding = self.data["embedding"]
        self.rnn = nn.RNN(self.embedding.embedding_dim, 256, batch_first=True)
        self.fc1 = nn.Linear(256, len(self.data["vocab"]))

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)

        hidden = hidden.squeeze(0)
        return self.fc1(hidden)

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        dataloader = self.data["data_loader"]

        for epoch in range(self.max_epoch):
            train_loss, y_pred, y_true = None, None, None
            avg_train_loss = 0

            for id_batch, (X_batch, y_batch) in tqdm(enumerate(dataloader), total=len(dataloader), leave=True, desc="Training"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = self.forward(X_batch)
                y_true = y_batch

                train_loss = loss_function(y_pred, y_true)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                self.losses.append(train_loss.item())
                avg_train_loss += train_loss.item()

            avg_train_loss /= len(dataloader)
            scheduler.step(avg_train_loss)

            accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self):
        self.eval()
        dataloader = self.data["data_loader"]
        y_pred = []
        y_true = []
        with torch.no_grad():
            for id_batch, (X_batch, y_batch) in enumerate(dataloader):
                y_pred_batch = self.forward(X_batch.to(device))
                y_pred.append(y_pred_batch.cpu())
                y_true.append(y_batch.cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        self.train()
        return y_pred.max(1)[1], y_true

    def run(self):
        self.build()
        self.to(device)

        print('method running...')
        print('--start training...')

        self.train_model()

        print('--end training...')

        pred_y, true_y  = self.test()

        return {"state_dict": self.state_dict(), "vocab": self.data["vocab"], "embedding": self.embedding,
                "pred_y": pred_y, "true_y": true_y, 'losses': self.losses, 'batches_per_epoch': len(self.data["data_loader"])}