import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch

dataset = pd.read_csv('../data/Iris.csv')
dataset = dataset.drop(['Id'], axis=1)

X = dataset.drop(['Species'], axis=1)
y = dataset['Species']

# encoder implementation
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

class IrisDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

