from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Dataset(Dataset):
    def __init__(self, data, label):
        self.x_data = data  # sent_pad / index 화된 data
        self.y_data = label  # 0,1 처리된 data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Variable(torch.LongTensor(self.x_data[idx]))
        y = Variable(torch.FloatTensor(np.expand_dims(self.y_data[idx], axis=0)))
        return x, y

def split_data(sent_pad, label):
    X, X_test, y, y_test = train_test_split(sent_pad, label, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = Dataset(X_train, y_train)
    val_data = Dataset(X_val, y_val)
    test_data = Dataset(X_test, y_test)

    batch_size = 64

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader