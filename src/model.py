import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, ntag, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_emb = nn.Embedding(ntoken, ninp)
        self.tag_emb = nn.Embedding(ntag, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(2*ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(2*ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        
        self.w_tag_mid = nn.Linear(nhid, nhid)
        self.w_word_mid = nn.Linear(nhid, nhid)
        self.w_tag_out = nn.Linear(nhid, ntag)
        self.w_word_out = nn.Linear(nhid, ntoken)

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.tag_emb.weight.data.uniform_(-initrange, initrange)
        self.w_tag_mid.bias.data.fill_(0)
        self.w_tag_mid.weight.data.uniform_(-initrange, initrange)
        self.w_word_mid.bias.data.fill_(0)
        self.w_word_mid.weight.data.uniform_(-initrange, initrange)
        self.w_tag_out.bias.data.fill_(0)
        self.w_tag_out.weight.data.uniform_(-initrange, initrange)
        self.w_word_out.bias.data.fill_(0)
        self.w_word_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_token, input_tag, hidden):
        word_emb = self.drop(self.word_emb(input_token))  # [1, batch_size, emb_size]
        tag_emb = self.drop(self.tag_emb(input_tag))  # [1, batch_size, emb_size]
        emb = torch.cat((word_emb, tag_emb), 2)  # [1, batch_size, 2*emb_size]
        
        output, hidden = self.rnn(emb, hidden)  # hidden[0] = [num_layers, batch, hidden_size]
        output = self.drop(output)   # [1, batch_size, hidden_size]
        output = output.view(output.size(0)*output.size(1), output.size(2)) # [batch_size, hidden_size]
        
        h_tag = self.w_tag_mid(output)
        h_word = self.w_word_mid(output)
        p_tag = F.softmax(self.w_tag_out(h_tag), dim=1)
        h_word_final = self.w_word_out(h_tag + h_word)
        p_word = F.softmax(h_word_final, dim=1)
        return p_word, p_tag, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
