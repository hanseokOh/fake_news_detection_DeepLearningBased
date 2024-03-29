import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# import bcolz
import pickle

# Device Setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

torch.manual_seed(2019)
if device == 'cuda':
    torch.cuda.manual_seed_all(2019)

import matplotlib.pyplot as plt


dtype = torch.FloatTensor

def create_emb_layer(weights_matrix, non_trainable=False):
    vocab_size, d_embedding = weights_matrix.shape[0], weights_matrix.shape[1]
    emb_layer = nn.Embedding(vocab_size, d_embedding) # vocab_size, d_embedding
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix).type(torch.FloatTensor)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, vocab_size, d_embedding


'''수정 -저장할 파라미터 추가'''
def save_model(model, optimizer, epoch, train_loss,validation_loss, saved_dir, file_name='best_model.pt'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'total_train_loss':train_loss,
        'total_validation_loss': validation_loss
    }
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)


def accuracy(pred, target):
    pred_y = pred >= 0.5
#     print(pred_y.dtype)
    num_correct = target.eq(pred_y.float()).sum()
#     print(num_correct)
    accuracy = (num_correct.item() * 100.0 / len(target))
    return accuracy


def f1_measure(pred, target):
    pred_y = pred >= 0.5
    pred_y = pred_y.int()
    pred_y = pred_y.tolist()
    pred_y = [p[0] for p in pred_y] # get items of each element

    eps = 1e-8

    tp = sum([p == 1 and t == 1 for p, t in zip(pred_y, target)])
    tn = sum([p == 0 and t == 0 for p, t in zip(pred_y, target)])
    fp = sum([p == 1 and t == 0 for p, t in zip(pred_y, target)])
    fn = sum([p == 0 and t == 1 for p, t in zip(pred_y, target)])

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    return f1_score

'''수정 - train_loss / validation_loss 같이 plot'''
def plot_results(x, y, z):
    """
    plot the results
    """

    fig = plt.figure(figsize=(9, 6))
    plt.plot(x, y, z)
    plt.xlabel('Number of steps')
    plt.ylabel('loss')
    plt.legend(('train_loss', 'validation_loss'))
    plt.title("Loss over epoch")
    plt.show()

