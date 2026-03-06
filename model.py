import torch
from torch import nn
import torch.nn.functional as F
from custom_activation import SqeLon
from torch.optim import Adam

from dataset import train_loader, test_loader

class IrisModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisModel, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size)
        self.hl2 = nn.Linear(hidden_size, hidden_size)
        self.op = nn.Linear(hidden_size, num_classes)
        self.sqelon = SqeLon()

    def forward(self, x):
        # x = F.relu(self.hl1(x))
        x = self.sqelon(self.hl1(x))
        x = self.sqelon(self.hl2(x))
        x = self.op(x)

        return x

model = IrisModel(4, 32, 3)

optimization = Adam(model.parameters(), lr=0.01)
criteria = nn.CrossEntropyLoss()

num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimization.zero_grad()
        outputs = model(X_batch)
        loss = criteria(outputs, y_batch)
        loss.backward()
        optimization.step()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total = y_batch.size(0)
            correct += (predicted == y_batch).sum().item()


    print(f"Epoch: {epoch+1}, Accuracy: {correct/total:.3f}")