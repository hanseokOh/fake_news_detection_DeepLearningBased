from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pickle

class Dataset_BaseDL(Dataset):
    def __init__(self, data, label):
        self.x_data = data  # sent_pad / index 화된 data
        self.y_data = label  # 0,1 처리된 data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = Variable(torch.LongTensor(self.x_data[idx]))
        y = Variable(torch.FloatTensor(np.expand_dims(self.y_data[idx], axis=0)))
        return x, y


class Dataset_EAN(Dataset):
    def __init__(self, cls, kg, ent, labels):
        self.cls = cls
        self.kg = kg
        self.ent = ent
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X_1 = np.asarray(self.cls[idx])
        X_2 = np.asarray(self.kg[idx])
        X_3 = np.asarray(self.ent[idx])
        y = np.expand_dims(self.labels[idx], axis=0)  # [1,]
        sent_len = len(X_1)

        if sent_len < 64:
            pad_num = 64 - sent_len
            pad = np.zeros([pad_num, 768])
            X_1 = np.concatenate((X_1, pad), axis=0)
            X_2 = np.concatenate((X_2, pad), axis=0)
            X_3 = np.concatenate((X_3, pad), axis=0)
        X_4 = np.concatenate((X_2, X_3), axis=0)

        X_c = torch.FloatTensor(X_1)
        X_e = torch.FloatTensor(X_4)
        y = torch.FloatTensor(y)
        return X_c, X_e, y

def split_data_BaseDL(sent_pad_path, label_path):
    sent_pad = np.load(sent_pad_path, allow_pickle=True)
    label = pickle.load(open(label_path, 'rb'))

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


def split_data_EAN(cls_file_name, KG_file_name, context_file_name, labels_file_name, val_ratio):
    with open(cls_file_name, 'rb') as f:
        cls = pickle.load(f)

    with open(KG_file_name, 'rb') as f:
        kg = pickle.load(f)

    with open(context_file_name, 'rb') as f:
        ent = pickle.load(f)

    with open(labels_file_name, 'rb') as f:
        labels = pickle.load(f)

    cls_train, clscls_val, y_train, yy_val = train_test_split(cls,
                                                              labels,
                                                              test_size=2 * val_ratio,
                                                              random_state=42)

    cls_val, cls_test, y_val, y_test = train_test_split(clscls_val,
                                                        yy_val,
                                                        test_size=0.5,
                                                        random_state=42)

    kg_train, kgkg_val, ent_train, entent_val = train_test_split(kg,
                                                                 ent,
                                                                 test_size=2 * val_ratio,
                                                                 random_state=42)

    kg_val, kg_test, ent_val, ent_test = train_test_split(kgkg_val,
                                                          entent_val,
                                                          test_size=0.5,
                                                          random_state=42)

    return cls_train, cls_val, cls_test, kg_train, kg_val, kg_test, ent_train, ent_val, ent_test, y_train, y_val, y_test