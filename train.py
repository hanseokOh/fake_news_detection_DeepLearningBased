import argparse

import numpy as np
import torch

from model.bi_lstm import BiLSTM
from trainer.BaseDL import Trainer


def define_argparser():
    '''
    Define argument parser
    :return: configuration object
    '''

    # NOTE : We assume that the dataset is not separated.
    parser = argparse.ArgumentParser(description = 'run argparser')
    parser.add_argument('--model',required=False,default='bi-lstm', help='select model')
    parser.add_argument('--data_path',required=False,default = '', help='fake news data path (csv format), must include text, type columns')
    parser.add_argument('--weights_matrix',required=False,default = 'object/BiLSTM/weights_matrix_840B_300.npy', help='weights matrix path for word embeddings')
    parser.add_argument('--save_dir', required=True, help='where to save model checkpoint')

    parser.add_argument('--batch_size', type=int, default= 64)
    parser.add_argument('--n_epochs', type=int, default=10)

    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--n_class', type=int, default=1, help="We only implement binary classification case. Multiclass will be updated soon.")
    parser.add_argument('--learning rate', type=float, required=False, default=1e-5)

    parser.add_argument('--val_every', type=int, default=1)
    args = parser.parse_args()
    return args

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weights_matrix = np.load(args.weights_matrix, allow_pickle=True)  # 새로 저장

    if args.model == 'bi-lstm':
        model = BiLSTM(weights_matrix).to(device)
        model_path = ''

    # TO DO
    elif args.model == 'cnn':
        pass

    trainer = Trainer(args)
    trainer.train(num_epochs=args.n_epochs,
              model=model,
              saved_dir=args.save_dir,
              device=device,
              criterion=torch.nn.BCELoss(),
              optimizer=torch.optim.Adam(params=model.parameters(), lr = 1e-5),
              val_every=args.val_every
              )


if __name__ == '__main__':
    args = define_argparser()
    main(args)