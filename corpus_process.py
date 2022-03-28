# encoding=utf-8
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
from nltk.corpus import stopwords
from itertools import chain
import codecs
import json
import collections


def make_vector(texts, text_size, sent_len):  # Pad the conv/history with 0s to fixed size
    text_vec = []
    for one_text in texts:
        t = []
        for sent in one_text:
            pad_len = max(0, sent_len - len(sent))
            t.append(sent + [0] * pad_len)
        pad_size = max(0, text_size - len(t))
        t.extend([[0] * sent_len] * pad_size)
        text_vec.append(t)
    return torch.LongTensor(text_vec)


class MyDataset(data.Dataset):
    def __init__(self, corp, current_ids, history_size=30, pos_sample_weight=1, use_BERT=False, use_TDM=False):
        self.need_user_history = (True if history_size > 0 else False)
        self.use_BERT = use_BERT
        self.use_TDM = use_TDM
        if self.use_BERT:
            convs = corp.convs_tokens
            # user_history = corp.user_history_tokens
        else:
            convs = corp.convs_ids
        user_history = corp.user_history_ids
        data_conv = []
        data_label = []
        data_weight = []
        if self.need_user_history:
            self.history_size = min(history_size, max([len(user_history[u]) for u in user_history.keys()]) + 1)
            data_history = []
        for cid in current_ids:
            data_label.append(corp.labels[cid])
            if corp.labels[cid] == 1:
                data_weight.append(pos_sample_weight)
            else:
                data_weight.append(1)
            if self.use_BERT:
                data_conv.append([turn[1] for turn in convs[cid]])
            else:
                data_conv.append(convs[cid])
            if self.need_user_history:
                uid = convs[cid][-1][0]
                hist = user_history[uid][:self.history_size - 1]
                # add new user's current turn to history, avoiding history size to be 0
                if not self.use_BERT:
                    current_turn = [cid]
                    current_turn.extend(convs[cid][-1][1:])
                    hist.append(current_turn)
                data_history.append(hist)
        if self.use_BERT:
            self.data_conv = data_conv
            if self.need_user_history:
                self.data_history = data_history
        else:  # make torch vector
            self.conv_turn_size = max([len(c) for c in data_conv])
            self.conv_sent_len = max([len(sent) for sent in chain.from_iterable([c for c in data_conv])])
            self.data_conv = make_vector(data_conv, self.conv_turn_size, self.conv_sent_len)
            if self.need_user_history:
                self.history_sent_len = max([len(sent) for sent in chain.from_iterable([h for h in data_history])])
                self.data_history = make_vector(data_history, self.history_size, self.history_sent_len)
        self.data_label = torch.Tensor(data_label)
        self.data_weight = torch.Tensor(data_weight)

        if self.use_TDM:
            vocab_num = corp.wordNum
            convs = corp.convs_ids
            user_history = corp.user_history_ids
            self.data_history_BOW = []
            self.data_conv_BOW = []
            for cid in current_ids:
                current_conv = []
                for turn in convs[cid]:
                    current_turn = np.zeros(vocab_num, dtype=np.int32)
                    for word in turn[1:]:
                        current_turn[word] += 1
                    current_conv.append(current_turn)
                self.data_conv_BOW.append(current_conv)
                if self.need_user_history:
                    uid = convs[cid][-1][0]
                    current_hist = []
                    for hist in user_history[uid][:self.history_size - 1]:
                        current_turn = np.zeros(vocab_num, dtype=np.int32)
                        for word in hist[1:]:
                            current_turn[word] += 1
                        current_turn_conv_content = convs[hist[0]]
                        current_turn_conv = []
                        for turn in current_turn_conv_content:
                            current_turn_turn = np.zeros(vocab_num, dtype=np.int32)
                            for word in turn[1:]:
                                current_turn_turn[word] += 1
                            current_turn_conv.append(current_turn)
                        current_hist.append([current_turn, current_turn_conv])  # each hist stores hist turn and its context
                    current_hist.append([current_conv[-1], current_conv])
                    self.data_history_BOW.append(current_hist)

    def __getitem__(self, idx):
        if self.use_TDM:
            return self.data_conv[idx], self.data_conv_BOW[idx], self.data_history_BOW[idx], self.data_label[idx]
        elif self.need_user_history:
            return self.data_conv[idx], self.data_history[idx], self.data_label[idx]
        else:
            return self.data_conv[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_label)

    def collate_fn(self, batch):
        ll = len(batch[0])
        batches = [[] for i in range(ll)]
        for item in batch:
            for n in range(ll):
                batches[n].append(item[n])
        batches[-1] = torch.Tensor(batches[-1])
        return batches


class Corpus:

    def __init__(self, trainfile, testfile, validfile, batch_size, history_size=30, use_BERT=False, use_TDM=False):

        self.convNum = 0            # Number of conversations
        self.convIDs = {}           # Dictionary that maps conversations to integer IDs
        self.r_convIDs = {}         # Inverse of last dictionary
        self.userNum = 0            # Number of users
        self.userIDs = {}           # Dictionary that maps users to integer IDs
        self.r_userIDs = {}         # Inverse of last dictionary
        self.wordNum = 1            # Number of words
        self.wordIDs = {'<ZeroPad>': 0}           # Dictionary that maps words to integers
        self.r_wordIDs = {0: '<ZeroPad>'}         # Inverse of last dictionary
        self.msgNum = 0             # Number of messages

        # Each conv is a list of turns, each turn is [userID, w1, w2, w3, ...]
        self.convs_ids = collections.defaultdict(list)
        self.convs_tokens = collections.defaultdict(list)
        self.convs_in_train = set()
        self.convs_in_test = set()
        self.convs_in_valid = set()
        # Store each conv's label, 1 or 0
        self.labels = {}
        # Stores each user's history message, each message is [convID, w1, w2, w3, ...]
        self.user_history_ids = collections.defaultdict(list)
        # self.user_history_tokens = collections.defaultdict(list)
        # print(use_BERT, use_TDM)

        wordCount = collections.Counter()  # The count every word appears

        cur_userNum, cur_convNum, cur_msgNum = 0, 0, 0
        for filename in [trainfile, testfile, validfile]:
            with codecs.open(filename, 'r', 'utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    msgs = json.loads(line)
                    conv_id = msgs[0][0][0]
                    if 'train' in filename:
                        self.convs_in_train.add(self.convNum)
                    elif 'test' in filename:
                        self.convs_in_test.add(self.convNum)
                    else:
                        self.convs_in_valid.add(self.convNum)
                    self.convIDs[conv_id] = self.convNum
                    self.r_convIDs[self.convNum] = conv_id
                    self.labels[self.convNum] = msgs[-1]
                    self.convNum += 1
                    for turn in msgs[0]:
                        self.msgNum += 1
                        user_id = turn[2]
                        if user_id not in self.userIDs:
                            self.userIDs[user_id] = self.userNum
                            self.r_userIDs[self.userNum] = user_id
                            self.userNum += 1
                        words = []
                        for word in turn[3].split(' '):
                            if wordCount[word] == 0:
                                self.wordIDs[word] = self.wordNum
                                self.r_wordIDs[self.wordNum] = word
                                self.wordNum += 1
                            words.append(self.wordIDs[word])
                            wordCount[word] += 1
                        tokens = '[CLS] ' + turn[3].strip()
                        current_turn = [self.userIDs[user_id]]
                        current_turn.extend(words)
                        self.convs_ids[self.convIDs[conv_id]].append(current_turn)
                        self.convs_tokens[self.convIDs[conv_id]].append([self.userIDs[user_id], tokens])
                        if 'train' in filename:
                            current_msg = [self.convIDs[conv_id]]
                            current_msg.extend(words)
                            self.user_history_ids[self.userIDs[user_id]].append(current_msg)
                            # self.user_history_tokens[self.userIDs[user_id]].append(tokens)
            if 'train' in filename:
                print("%s process over! UserNum: %d ConvNum: %d MsgNum: %d" % (filename, self.userNum-cur_userNum, self.convNum-cur_convNum, self.msgNum-cur_msgNum))
            else:
                print("%s process over! UserNum(Not in train): %d ConvNum: %d MsgNum: %d" % (filename, self.userNum-cur_userNum, self.convNum-cur_convNum, self.msgNum-cur_msgNum))
            cur_userNum, cur_convNum, cur_msgNum = self.userNum, self.convNum, self.msgNum

        self.stopwordIDs = {}
        mystopwords = set(stopwords.words('english'))
        for w in self.wordIDs.keys():
            if self.wordIDs[w] in mystopwords:
                self.stopwordIDs[w] = self.wordIDs[w]

        self.convs_in_train = [cid for cid in self.convs_in_train]
        self.convs_in_test = [cid for cid in self.convs_in_test]
        self.convs_in_valid = [cid for cid in self.convs_in_valid]

        # self.train_data = MyDataset(self, self.convs_in_train, history_size, use_BERT=use_BERT, use_TDM=use_TDM)
        # self.train_loader = data.DataLoader(self.train_data, collate_fn=self.train_data.collate_fn, batch_size=batch_size, num_workers=1, shuffle=True)
        # print(use_BERT, use_TDM, 'in corpus')
        # exit()
        self.test_data = MyDataset(self, self.convs_in_test, history_size, pos_sample_weight=1, use_BERT=use_BERT, use_TDM=use_TDM)
        self.test_loader = data.DataLoader(self.test_data, collate_fn=self.test_data.collate_fn, batch_size=batch_size, num_workers=0, shuffle=True)
        self.valid_data = MyDataset(self, self.convs_in_valid, history_size, pos_sample_weight=1, use_BERT=use_BERT, use_TDM=use_TDM)
        self.valid_loader = data.DataLoader(self.valid_data, collate_fn=self.valid_data.collate_fn, batch_size=batch_size, num_workers=0, shuffle=True)

        print("Corpus process over! Overall statistic: UserNum: %d ConvNum: %d MsgNum: %d" % (self.userNum, self.convNum, self.msgNum))


def create_embedding_matrix(dataname, word_idx, word_num, embedding_dim=200):
    pretrain_file = '../LSTMBiA/glove.twitter.27B.200d.txt' if dataname[0] == 't' else '../LSTMBiA/glove.6B.200d.txt'
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    weights_matrix = np.zeros((word_num, embedding_dim))
    for idx in word_idx.keys():
        if idx == 0:
            continue
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)


if __name__ == '__main__':
    corp = Corpus('tw_train.json', 'tw_test.json', 'tw_valid.json', 32, 30)
    # print corp.train_data.data_conv.size()
    # print corp.train_data.data_history.size()




