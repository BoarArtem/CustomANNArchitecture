import torch
from torch import nn
from torch.optim import Adam
from custom_activation import SqeLon
from dataset import train_loader, test_loader
from test import test
from train import train
import logging

logger = logging.getLogger(__name__)

class IrisModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.op = nn.Linear(hidden_size, num_classes)

        self.sqelon = SqeLon()

    def forward(self, x):
        x = self.sqelon(self.fc1(x))
        x = self.sqelon(self.fc2(x))
        x = self.op(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = IrisModel(input_size=4, hidden_size=64, num_classes=3).to(device)

# optimization and loss function
optim = Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

if __name__ == "__main__":
    logger.info("Training is started...")
    train(model, 100, train_loader, optim, criterion, device)
    test(model, test_loader, device)