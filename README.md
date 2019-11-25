# Fake News Detection DeepLearning Based Model
-----------------

This repo is a secondary repo for [fake_news_TNA_SC](https://github.com/jucho2725/fake_news_detection_TNA_SC). Specially focused on Deep Learning based model: LSTM based, CNN based and Transformer based model.

## Pre-requisite
-------------------------
- Python 3.6.8
- Pytorch 1.3.1
- [glove 6B - 300 dimension embedding](https://nlp.stanford.edu/projects/glove/)
- index table about glove embedding
- preprocessed data : use proper padding according to model


## Usage
-----------------------

### Preparation

#### Format

#### Tokenization and padding

### Train

<pre><code>
$python train.py -h
usage: train.py [-h] [--model MODEL] [--data_path DATA_PATH]
                [--weights_matrix WEIGHTS_MATRIX] --save_dir SAVE_DIR
                [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS]
                [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS]
                [--n_class N_CLASS] [--learning rate LEARNING RATE]
                [--val_every VAL_EVERY]
                
or you can check details on train.py file

we don't provide 'object/BiLSTM/weights_matrix_840B_300.npy'

(in progress: change data_path parser and insert options for other models e.g. CNN, Tranformer )

def define_argparser():
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

</pre></code>



### Inference

<pre><code>
$python classify.py -h
usage: classify.py [-h] [--model MODEL] [--weights_matrix WEIGHTS_MATRIX]
                   [--model_path MODEL_PATH] [--sent_pad_path SENT_PAD_PATH]
                   [--label_path LABEL_PATH] [--data_path DATA_PATH]

or you can check details on classify.py file

we don't provide 'weights_matrix_6B_300.npy, best_model.pt, sent_pad_modified.npy, label_modified.pkl' files.

def define_argparser():
    # argparse
    parser = argparse.ArgumentParser(description = 'run argparser')
    parser.add_argument('--model',required=False,default='bi-lstm', help='select model')
    parser.add_argument('--weights_matrix',required=False,default = 'data/weights_matrix_6B_300.npy', help='weights matrix path for         word embeddings')
    parser.add_argument('--model_path',required=False, default = 'data/best_model.pt', help='model checkpoint path')

    parser.add_argument('--sent_pad_path',required=False, default ='data/sent_pad_modified.npy', help='padded sentence(preprocessed)')
    parser.add_argument('--label_path',default='data/label_modified.pkl')
    parser.add_argument('--data_path',required=False, default = '', help='fake news data path (csv format), must include text, type         columns')

    args = parser.parse_args()
    return args

                   


</code></pre>

