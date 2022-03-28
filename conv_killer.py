import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class ConvKiller(nn.Module):
    def __init__(self, config, max_turn_num, max_turn_len):
        super(ConvKiller, self).__init__()
        self.max_turn_num = max_turn_num
        self.max_turn_len = max_turn_len
        self.hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size
        self.turn_num_embedding = nn.Embedding(self.max_turn_num+1, 20, padding_idx=0)
        self.turn_len_embedding = nn.Embedding(self.max_turn_len+1, 20, padding_idx=0)
        self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
        self.conv_lstm = nn.LSTM(config.embedding_dim+40, self.hidden_dim, num_layers=2, dropout=config.dropout, bidirectional=True)
        self.layernorm = nn.LayerNorm(self.hidden_dim*2)
        self.att_weight = nn.Parameter(torch.randn((self.max_turn_num, self.max_turn_num)))
        self.mlp1 = nn.Linear(self.hidden_dim*4, self.hidden_dim*2)
        self.mlp2 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.final = nn.Sigmoid()

    def forward(self, convs):
        # print(self.max_turn_len, self.max_turn_num)
        conv_reps = []
        conv_turn_nums = []
        for conv in convs:
            turn_num = torch.LongTensor([(conv[:, 1] > 0).sum()])
            turn_lens = torch.LongTensor([(turn[1:] > 0).sum() for turn in conv])
            if torch.cuda.is_available():  # run in GPU
                conv = conv.cuda()
                turn_num = turn_num.cuda()
                turn_lens = turn_lens.cuda()
            turn_reps = self.word_embedding(conv[:, 1:])
            # print(turn_lens, turn_num)
            turn_len_reps = self.turn_len_embedding(turn_lens)
            turn_num_reps = self.turn_num_embedding(turn_num).repeat(conv.size(0), 1)
            # print('turn', turn_reps.size(), turn_num_reps.size(), turn_len_reps.size())
            turn_reps = turn_reps.sum(dim=1) / turn_lens.unsqueeze(-1).float()
            turn_reps = torch.cat([turn_reps, turn_len_reps, turn_num_reps], dim=1)
            conv_reps.append(turn_reps)
            conv_turn_nums.append(turn_num)
        conv_turn_nums = torch.cat(conv_turn_nums)
        # print('cturn', conv_turn_nums)
        sorted_conv_turn_nums, indices = torch.sort(conv_turn_nums, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_conv_reps = [conv_reps[index] for index in indices]
        paded_convs = rnn_utils.pad_sequence(sorted_conv_reps)
        # print('conv', paded_convs.size())
        # print(paded_convs.size())
        # print(sorted_conv_turn_nums)
        packed_convs = rnn_utils.pack_padded_sequence(paded_convs, sorted_conv_turn_nums)
        lstm_out, lstm_hidden = self.conv_lstm(packed_convs)
        lstm_hidden = torch.cat([lstm_hidden[0][-2], lstm_hidden[0][-1]], dim=1)[desorted_indices]
        lstm_out = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)[0][desorted_indices]
        lstm_hidden = self.layernorm(lstm_hidden)
        lstm_out = self.layernorm(lstm_out)
        # print(conv_turn_nums, self.att_weight.size(), lstm_out.size())
        att_weight = torch.cat([self.att_weight[tnum-1, :lstm_out.size(1)].unsqueeze(0) for tnum in conv_turn_nums], dim=0)
        att_weight = F.softmax(att_weight, dim=1)
        # print(att_weight.size(), lstm_out.size())
        att_out = (att_weight.unsqueeze(-1) * lstm_out).sum(dim=1)
        final_out = self.tanh(self.mlp1(torch.cat([lstm_hidden, att_out], dim=1)))
        final_out = self.relu(self.mlp2(final_out))
        conv_labels = self.final(self.hidden2label(final_out).view(-1))
        return conv_labels

