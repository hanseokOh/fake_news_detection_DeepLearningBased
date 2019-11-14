

class BiLSTM(nn.Module):
    def __init__(self, weights_matrix, hidden_size=100, num_layers=3, n_class=1, dropout=0.1):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding, vocab_size, d_embedding = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(input_size=d_embedding, hidden_size=hidden_size, dropout=dropout,
                            num_layers=num_layers, bidirectional=True, batch_first=True)
        self.generator1 = nn.Linear(hidden_size * 2, 10)
        self.generator2 = nn.Linear(10, n_class)
        #         self.activation = nn.LogSoftmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, x):  # x = list(각 문서 리스트. 단어들이 token index로 표현되어있음) of list
        hidden, cell = self.init_hidden(x.size(0))
        h, c = hidden.to(x.device), cell.to(x.device)
        # |x| = (batch_size, length)
        x = self.embedding(x)
        # |x| = (batch_size, length, d_embedding)
        x, _ = self.lstm(x, (h, c))
        # |x| = (batch_size, length, hidden_size * 2) # bidrectional 하므로 hidden size가 두 배
        # |x[:, -1]| = (batch_size, length * hidden_size * 2) # flattened
        x = torch.cat((x[:, 1, :self.hidden_size], x[:, 0, self.hidden_size:]), 1)
        x = self.generator1(x)
        x = self.generator2(x)
        # |y| = (batch_size, n_classes)
        y = self.activation(x)
        return y, (h, c)

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        return hidden, cell



