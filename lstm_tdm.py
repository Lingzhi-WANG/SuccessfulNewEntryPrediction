import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import math
import torch.nn.utils.rnn as rnn_utils
from TDMmodel import conv_models, criterions, encoders, model_bases, utils
# from transformers import BertTokenizer, BertModel, BertForMaskedLM


# class SentBERT:
#     def __init__(self):
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.BERTmodel = BertModel.from_pretrained('bert-base-uncased')
#         self.BERTmodel.eval()
#         # for param in self.BERTmodel.parameters():
#         #   param.requires_grad = False
#         if torch.cuda.is_available():  # run in GPU
#             self.BERTmodel = self.BERTmodel.to('cuda')
#
#     def sent_vector(self, conv):
#
#         conv_tokens_tensor = []
#
#         for turn in conv:
#             turn_tokenized_text = self.tokenizer.tokenize(turn)
#             turn_indexed_tokens = self.tokenizer.convert_tokens_to_ids(turn_tokenized_text)
#             turn_segments_ids = [0 for i in turn_indexed_tokens]
#             tokens_tensor = torch.Tensor([turn_indexed_tokens])
#             segments_tensors = torch.Tensor([turn_segments_ids])
#             tokens_tensor = tokens_tensor.to(torch.int64).to('cuda')
#             segments_tensors = segments_tensors.to(torch.int64).to('cuda')
#             with torch.no_grad():
#                 outputs = self.BERTmodel(tokens_tensor, token_type_ids=segments_tensors)
#                 predictions = outputs[0][0][0]
#                 conv_tokens_tensor.append(predictions.view(1, -1))
#         # print(conv_tokens_tensor[0].size())
#         # print(torch.cat(conv_tokens_tensor, dim=0).size())
#         return torch.cat(conv_tokens_tensor, dim=0)


class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim, kernel_num, dropout):
        super(CNNEncoder, self).__init__()
        self.kernel_num = kernel_num
        self.cnn1 = nn.Conv2d(1, self.kernel_num, (2, embedding_dim))
        self.cnn2 = nn.Conv2d(1, self.kernel_num, (3, embedding_dim))
        self.cnn3 = nn.Conv2d(1, self.kernel_num, (4, embedding_dim))
        self.dropout = nn.Dropout(dropout)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        # print x.size()
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, sentences):
        embeds = sentences.unsqueeze(1)
        cnn1_out = self.conv_and_pool(embeds, self.cnn1)
        cnn2_out = self.conv_and_pool(embeds, self.cnn2)
        cnn3_out = self.conv_and_pool(embeds, self.cnn3)
        sent_reps = torch.cat([cnn1_out, cnn2_out, cnn3_out], dim=1)
        sent_reps = self.dropout(sent_reps)
        return sent_reps


class LSTMEncoder(nn.Module):
    def __init__(self, config, input_dim, hidden_dim):
        super(LSTMEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.topic_init = config.topic_init
        self.lstm = nn.GRU(self.input_dim, self.hidden_dim, dropout=config.dropout, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sentences, sentence_length, *topic_vectors):

        sorted_sentence_length, indices = torch.sort(sentence_length, descending=True) 
        sorted_sentences = sentences[indices]
        _, desorted_indices = torch.sort(indices, descending=False)
        packed_sentences = rnn_utils.pack_padded_sequence(sorted_sentences, sorted_sentence_length, batch_first=True)
        if self.topic_init:
            sorted_topics = topic_vectors[0][indices]
            hidden = torch.cat([sorted_topics.unsqueeze(0), sorted_topics.unsqueeze(0)], dim=0)
            _, lstm_output = self.lstm(packed_sentences, hidden)
        else:
            _, lstm_output = self.lstm(packed_sentences)
        lstm_output = torch.cat([lstm_output[-1], lstm_output[-2]], dim=-1)[desorted_indices]
        lstm_output = self.dropout(lstm_output)
        return lstm_output


class LSTMTDM(nn.Module):
    def __init__(self, config):
        super(LSTMTDM, self).__init__()
        self.conv_hidden_dim = config.hidden_dim
        self.batch_size = config.batch_size
        self.bi_direction = True
        self.num_layer = config.model_num_layer
        self.d = config.d
        self.k = config.k
        self.model_type = config.model_type
        self.use_CNN = config.use_CNN
        self.use_LSTM = config.use_LSTM
        self.dd_att = config.dd_att
        self.topic_init = config.topic_init
        self.disc_lstm = config.disc_lstm
        self.topic_concat = config.topic_concat
        if self.topic_init:
            self.topic2hidden = nn.Linear(self.k, self.conv_hidden_dim)
        if self.use_CNN:
            self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
            if config.embedding_matrix is not None:
                self.word_embedding.load_state_dict({'weight': config.embedding_matrix})
            self.sent_modeling = CNNEncoder(config.embedding_dim, config.CNN_kernal_num, config.dropout)
            self.conv_lstm = nn.LSTM(config.CNN_kernal_num*3, self.conv_hidden_dim, dropout=config.dropout, num_layers=self.num_layer, bidirectional=self.bi_direction)
        elif self.use_LSTM:
            self.word_embedding = nn.Embedding(config.vocab_num, config.embedding_dim, padding_idx=0)
            if config.embedding_matrix is not None:
                self.word_embedding.load_state_dict({'weight': config.embedding_matrix})
            self.sent_modeling = LSTMEncoder(config, config.embedding_dim, self.conv_hidden_dim)
            if self.disc_lstm:
                self.conv_lstm = nn.LSTM(self.conv_hidden_dim*2+self.d, self.conv_hidden_dim, dropout=config.dropout, num_layers=self.num_layer, bidirectional=self.bi_direction)
            else:
                self.conv_lstm = nn.LSTM(self.conv_hidden_dim*2, self.conv_hidden_dim, dropout=config.dropout, num_layers=self.num_layer, bidirectional=self.bi_direction)
        # else:
        #     self.sent_modeling = SentBERT()
        #     self.bert_to_hidden = nn.Linear(768, self.conv_hidden_dim)
        #     self.conv_lstm = nn.LSTM(self.conv_hidden_dim, self.conv_hidden_dim, dropout=config.dropout, num_layers=self.num_layer, bidirectional=self.bi_direction)
        # # self.conv_hidden = self.init_hidden(self.batch_size, self.num_layer, self.conv_hidden_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.final = nn.Sigmoid()
        if self.dd_att:
            self.dd_att_weight = nn.Parameter(torch.randn(self.d+1))
            out_dim = self.conv_hidden_dim * 4
        else:
            out_dim = self.conv_hidden_dim * 2
        if self.topic_concat:
            self.hidden2label = nn.Linear(out_dim + self.k, 1)
        else:
            self.hidden2label = nn.Linear(out_dim, 1)

    def init_hidden(self, batch_size, num_layer, hidden_dim):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi * num_layer, batch_size, hidden_dim).cuda(),
                    torch.randn(bi * num_layer, batch_size, hidden_dim).cuda())
        else:
            return (torch.randn(bi * num_layer, batch_size, hidden_dim),
                    torch.randn(bi * num_layer, batch_size, hidden_dim))

    def forward(self, convs, conv_TDvecs, history_TDvecs):
        batch_size = len(convs)
        conv_topics = conv_TDvecs[:, 0, self.d:]
        history_topics = history_TDvecs[:, :, self.d:]
        if self.topic_init:
            history_turn_num = (history_topics.sum(dim=-1) != 0).sum(dim=-1)
            avg_history_topics = history_topics.sum(dim=1) / history_turn_num.unsqueeze(-1).float()
        conv_reps = []
        conv_turn_nums = []
        c_num = 0
        for conv in convs:
            if self.use_CNN:
                if torch.cuda.is_available():  # run in GPU
                    conv = conv.cuda()
                turn_num = (conv[:, 1] > 0).sum()
                sent_reps = self.sent_modeling(self.word_embedding(conv[:turn_num, 1:]))
            elif self.use_LSTM:
                if torch.cuda.is_available(): 
                    conv = conv.cuda()
                turn_num = (conv[:, 1] > 0).sum()
                sentence_length = (conv[:, 1:] > 0).sum(dim=-1)
                if self.topic_init:
                    topic_vectors = conv_topics[c_num].unsqueeze(0).repeat(turn_num-1, 1)
                    topic_vectors = torch.cat([topic_vectors, avg_history_topics[c_num].unsqueeze(0)], dim=0)
                    topic_vectors = self.topic2hidden(topic_vectors)
                    sent_reps = self.sent_modeling(self.word_embedding(conv[:turn_num, 1:]), sentence_length[:turn_num], topic_vectors)
                else:
                    sent_reps = self.sent_modeling(self.word_embedding(conv[:turn_num, 1:]), sentence_length[:turn_num])

            else:
                turn_num = len(conv)
                sent_reps = self.sent_modeling.sent_vector(conv)
                sent_reps = self.tanh(self.bert_to_hidden(sent_reps))
            conv_reps.append(sent_reps)
            conv_turn_nums.append(turn_num)
            c_num += 1
        sorted_conv_turn_nums, indices = torch.sort(torch.LongTensor(conv_turn_nums), descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_conv_reps = []
        for index in indices:
            sorted_conv_reps.append(conv_reps[index])
        paded_convs = rnn_utils.pad_sequence(sorted_conv_reps, batch_first=True)
        if self.disc_lstm:
            sorted_conv_discourse = conv_TDvecs[:, :, :self.d][indices]
            paded_convs = torch.cat([paded_convs, sorted_conv_discourse], dim=2)
        packed_convs = rnn_utils.pack_padded_sequence(paded_convs, sorted_conv_turn_nums, batch_first=True)
        conv_hidden = self.init_hidden(batch_size, self.num_layer, self.conv_hidden_dim)
        lstm_out, conv_hidden = self.conv_lstm(packed_convs, conv_hidden)

        lstm_discourse_out = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)[0][desorted_indices]
        query_turn = torch.cat([conv_hidden[0][-1], conv_hidden[0][-2]], dim=1)[desorted_indices]

        # attention part
        if torch.cuda.is_available():  # run in GPU
            masks = torch.where(lstm_discourse_out[:, :, 0] > 0, torch.Tensor([0.]).cuda(), torch.Tensor([-np.inf]).cuda())
        else:
            masks = torch.where(lstm_discourse_out[:, :, 0] > 0, torch.Tensor([0.]), torch.Tensor([-np.inf]))
        if self.dd_att:
            assigned_discourse = torch.argmax(conv_TDvecs[:, :, :self.d], dim=2)
            att_weights = self.dd_att_weight[assigned_discourse]
            att_weights = att_weights[:, :lstm_discourse_out.size(1)] + masks
            for bs in range(batch_size):
                att_weights[bs, conv_turn_nums[bs] - 1] *= (math.cos(self.dd_att_weight[-1]) + 1)
            exp_att_weights = torch.exp(att_weights)
            expsum_att_weights = torch.sum(exp_att_weights, dim=1)
            att_weights = exp_att_weights / torch.clamp(expsum_att_weights.unsqueeze(1), min=1e-15)
            att_out = torch.bmm(lstm_discourse_out.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
            att_out = torch.cat([query_turn, att_out], dim=1)
        else:
            att_out = query_turn
            # for bs in range(batch_size):
            #     masks[bs, conv_turn_nums[bs] - 1] = torch.Tensor([-np.inf])
            # att_weights = (lstm_discourse_out * query_turn.unsqueeze(1)).sum(-1) + masks
            # # att_weights = att_weights - att_weights.max(dim=1)[0].unsqueeze(-1)
            # exp_att_weights = torch.exp(att_weights)
            # expsum_att_weights = torch.sum(exp_att_weights, dim=1)
            # att_weights = exp_att_weights / torch.clamp(expsum_att_weights.unsqueeze(1), min=1e-15)
            # # att_weights = F.softmax(att_weights - att_weights.max(dim=1)[0].unsqueeze(-1), dim=1)
            # att_out = torch.bmm(lstm_discourse_out.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
        # modeling topic similarity
        if self.topic_concat:
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            topic_similarity = cos(conv_topics.unsqueeze(1), history_topics)
            topic_similarity = F.softmax(topic_similarity, dim=1)
            att_history_topics = (topic_similarity.unsqueeze(-1) * history_topics).sum(dim=1)
            att_topics = att_history_topics
            final_out = torch.cat([att_out, att_topics], dim=1)
        else:
            final_out = att_out
        conv_labels = self.final(self.hidden2label(final_out).view(-1))

        return conv_labels





