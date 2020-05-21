import torch
import torch.nn as nn
import numpy as np

# Generate random data
# batch, seq_len, input_dim
rnn_input = torch.randn(70, 1000, 20)

unit = 'GRU'
input_dim = 20
hidden_dim = 128
batch_size = 70
n_layers = 2
drop_prob = 0.5

class RNN_NET(nn.Module):
    def __init__(self, unit, input_dim, hidden_dim, n_layers, drop_prob):
        super(RNN_NET, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        if unit == 'LSTM':
            self.rnn_unit = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = drop_prob, bidirectional = True)
        if unit == 'GRU':
            self.rnn_unit = nn.GRU(input_dim, self.hidden_dim, self.n_layers, batch_first = True, dropout = drop_prob, bidirectional = True)

        for weight in self.rnn_unit.parameters():
            if len(weight.size()) > 1:
                torch.nn.init.xavier_uniform_(weight)

    def forward(self, unit, rnn_input, hidden):
        p1_out, p1_hidden = self.rnn_unit(rnn_input, hidden)
        return p1_out

    def init_hidden(self, unit, batch_size):
        weight = next(self.parameters()).data
        if unit == 'LSTM':
            hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        if unit == 'GRU':
            hidden = weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_()
        return hidden

model = RNN_NET(unit, input_dim, hidden_dim, n_layers, drop_prob)
h = model.init_hidden(unit, batch_size)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Num of params: ", params)

rnn_output = model(unit, rnn_input, h)
print(rnn_output.size()) # 70, 1000, 256 
