import time
import torch
import random
import numpy as np
from itertools import chain
from torch import nn, optim
import torch.utils.data as data
import torch.nn.functional as F
from corpus_process import Corpus, create_embedding_matrix, MyDataset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(torch.clamp(output, min=1e-10, max=1))) + \
            weights[0] * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1)))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-10, max=1)) + \
               (1 - target) * torch.log(torch.clamp(1 - output, min=1e-10, max=1))

    return torch.neg(torch.mean(loss))


def joint_evaluate(model, TDMmodel, test_data, threshold=-1, compute_loss=None):  # evaluation metrics
    model.eval()
    TDMmodel.eval()
    true_labels = []
    pred_labels = []
    avg_loss = 0.0
    for step, one_data in enumerate(test_data):
        #print(one_data)
        #exit()
        vocab_size = len(one_data[1][0][0])
        # process data for modeling conversation turns with TDM
        turn_nums = [len(turn) for turn in one_data[1]]
        TDM_batch_size = sum(turn_nums)
        conv_context = np.zeros((TDM_batch_size, max(turn_nums), vocab_size), dtype=np.int32)
        conv_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
        for c in range(len(one_data[1])):
            for t in range(len(one_data[1][c])):
                b_pos = t + sum(turn_nums[:c])
                conv_targets[b_pos, :] = one_data[1][c][t]
                for i, row in enumerate(one_data[1][c]):
                    conv_context[b_pos][i][:] = row
        TDM_loss1, TDvecs1 = TDMmodel(conv_context, conv_targets, return_latent=True)
        conv_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs1.size(-1)))
        if torch.cuda.is_available():  # run in GPU
            conv_TDvecs = conv_TDvecs.cuda()
        for c in range(len(one_data[1])):
            for t in range(len(one_data[1][c])):
                b_pos = t + sum(turn_nums[:c])
                conv_TDvecs[c, t, :] = TDvecs1[b_pos, :]
        # process data for modeling history turns with TDM
        turn_nums = [len(hist) for hist in one_data[2]]
        max_context_num = max([len(turn[1]) for turn in chain.from_iterable([h for h in one_data[2]])])
        TDM_batch_size = sum(turn_nums)
        hist_context = np.zeros((TDM_batch_size, max_context_num, vocab_size), dtype=np.int32)
        hist_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
        for u in range(len(one_data[2])):
            for t in range(len(one_data[2][u])):
                b_pos = t + sum(turn_nums[:u])
                hist_targets[b_pos, :] = one_data[2][u][t][0]
                for i, row in enumerate(one_data[2][u][t][1]):
                    hist_context[b_pos][i][:] = row
        TDM_loss2, TDvecs2 = TDMmodel(hist_context, hist_targets, return_latent=True)
        hist_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs2.size(-1)))
        if torch.cuda.is_available():  # run in GPU
            hist_TDvecs = hist_TDvecs.cuda()
        for u in range(len(one_data[2])):
            for t in range(len(one_data[2][u])):
                b_pos = t + sum(turn_nums[:u])
                hist_TDvecs[u, t, :] = TDvecs2[b_pos, :]
        # produce predictions
        pred_label = model(one_data[0], conv_TDvecs, hist_TDvecs)
        label = one_data[-1]
        if compute_loss is not None:
            if torch.cuda.is_available():  # run in GPU
                label = label.cuda()
            avg_loss += weighted_binary_cross_entropy(pred_label, label, compute_loss).item()
        if torch.cuda.is_available():  # run in GPU
            pred_label = pred_label.cpu()
        pred_label = pred_label.data.numpy()
        label = one_data[-1].data.numpy()
        true_labels = np.concatenate([true_labels, label])
        pred_labels = np.concatenate([pred_labels, pred_label])

    try:
        auc = roc_auc_score(true_labels, pred_labels)
    except ValueError:
        auc = 0.0
    if compute_loss is not None:
        avg_loss /= len(test_data)
        return auc, avg_loss
    if threshold == -1:
        thr = 0.1
        best_thr = -1.0
        best_fc = -1.0
        while thr <= 0.9:
            current_pred_labels = (pred_labels >= thr)
            fc = f1_score(true_labels, current_pred_labels)
            if fc > best_fc:
                best_fc = fc
                best_thr = thr
            thr += 0.05
        threshold = best_thr
    current_pred_labels = (pred_labels >= threshold)
    acc = accuracy_score(true_labels, current_pred_labels)
    pre = precision_score(true_labels, current_pred_labels)
    rec = recall_score(true_labels, current_pred_labels)
    fc = (0 if pre == rec == 0 else 2 * pre * rec / (pre + rec))
    return acc, fc, pre, rec, auc, threshold


def joint_loss_compute(model, TDMmodel, using_data, loss_weights, TDM_weight=1):
    model.eval()
    TDMmodel.eval()
    avg_loss = 0.0
    for step, one_data in enumerate(using_data):
        label = one_data[-1]
        if torch.cuda.is_available():  # run in GPU
            label = label.cuda()
        vocab_size = len(one_data[1][0][0])
        # process data for modeling conversation turns with TDM
        turn_nums = [len(turn) for turn in one_data[1]]
        TDM_batch_size = sum(turn_nums)
        conv_context = np.zeros((TDM_batch_size, max(turn_nums), vocab_size), dtype=np.int32)
        conv_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
        for c in range(len(one_data[1])):
            for t in range(len(one_data[1][c])):
                b_pos = t + sum(turn_nums[:c])
                conv_targets[b_pos, :] = one_data[1][c][t]
                for i, row in enumerate(one_data[1][c]):
                    conv_context[b_pos][i][:] = row
        TDM_loss1, TDvecs1 = TDMmodel(conv_context, conv_targets, return_latent=True)
        conv_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs1.size(-1)))
        if torch.cuda.is_available():  # run in GPU
            conv_TDvecs = conv_TDvecs.cuda()
        for c in range(len(one_data[1])):
            for t in range(len(one_data[1][c])):
                b_pos = t + sum(turn_nums[:c])
                conv_TDvecs[c, t, :] = TDvecs1[b_pos, :]
        # process data for modeling history turns with TDM
        turn_nums = [len(hist) for hist in one_data[2]]
        max_context_num = max([len(turn[1]) for turn in chain.from_iterable([h for h in one_data[2]])])
        TDM_batch_size = sum(turn_nums)
        hist_context = np.zeros((TDM_batch_size, max_context_num, vocab_size), dtype=np.int32)
        hist_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
        for u in range(len(one_data[2])):
            for t in range(len(one_data[2][u])):
                b_pos = t + sum(turn_nums[:u])
                hist_targets[b_pos, :] = one_data[2][u][t][0]
                for i, row in enumerate(one_data[2][u][t][1]):
                    hist_context[b_pos][i][:] = row
        TDM_loss2, TDvecs2 = TDMmodel(hist_context, hist_targets, return_latent=True)
        hist_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs2.size(-1)))
        if torch.cuda.is_available():  # run in GPU
            hist_TDvecs = hist_TDvecs.cuda()
        for u in range(len(one_data[2])):
            for t in range(len(one_data[2][u])):
                b_pos = t + sum(turn_nums[:u])
                hist_TDvecs[u, t, :] = TDvecs2[b_pos, :]
        # produce predictions
        predictions = model(one_data[0], conv_TDvecs, hist_TDvecs)
        bce_loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
        avg_loss += bce_loss.item() + (TDM_loss1.item() + TDM_loss2.item()) * TDM_weight
    avg_loss /= len(using_data)
    return avg_loss


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfix_model(model):
    for param in model.parameters():
        param.requires_grad = True


def joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer, epoch, config, train_style, clip=0):
    start = time.time()
    print('%s Train Epoch: %d start!' % (train_style, epoch))
    if train_style == 'onlyTDM':
        TDMmodel.train()
        model.eval()
        fix_model(model)
        unfix_model(TDMmodel)
    elif train_style == 'fixTDM':
        model.train()
        TDMmodel.eval()
        unfix_model(model)
        fix_model(TDMmodel)
    else:
        TDMmodel.train()
        model.train()
        unfix_model(model)
        unfix_model(TDMmodel)
    random.shuffle(corp.convs_in_train)
    train_len = len(corp.convs_in_train)
    begin_idx = 0
    # count = 0
    avg_loss = 0.0
    while begin_idx < train_len:
        end_idx = begin_idx + config.batch_size * 100 if begin_idx + config.batch_size * 100 < train_len else train_len
        train_data = MyDataset(corp, corp.convs_in_train[begin_idx: end_idx], config.history_size, config.positive_sample_weight, use_BERT=False, use_TDM=True)
        sampler = data.WeightedRandomSampler(train_data.data_weight, num_samples=len(train_data), replacement=True)
        train_loader = data.DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=config.batch_size, num_workers=0, sampler=sampler)
        prob = (0.167 if config.filename == 'reddit' else 0.333)
        for step, one_data in enumerate(train_loader):
            tmp = random.random()
            # print(tmp, prob)
            if tmp > prob:
                continue
            label = one_data[-1]
            if torch.cuda.is_available():  # run in GPU
                label = label.cuda()
            vocab_size = len(one_data[1][0][0])
            # process data for modeling conversation turns with TDM
            turn_nums = [len(conv) for conv in one_data[1]]
            TDM_batch_size = sum(turn_nums)
            conv_context = np.zeros((TDM_batch_size, max(turn_nums), vocab_size), dtype=np.int32)
            conv_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
            for c in range(len(one_data[1])):
                for t in range(len(one_data[1][c])):
                    b_pos = t + sum(turn_nums[:c])
                    conv_targets[b_pos, :] = one_data[1][c][t]
                    for i, row in enumerate(one_data[1][c]):
                        conv_context[b_pos][i][:] = row
            TDM_loss1, TDvecs1 = TDMmodel(conv_context, conv_targets, return_latent=True)
            conv_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs1.size(-1)))
            if torch.cuda.is_available():  # run in GPU
                conv_TDvecs = conv_TDvecs.cuda()
            for c in range(len(one_data[1])):
                for t in range(len(one_data[1][c])):
                    b_pos = t + sum(turn_nums[:c])
                    conv_TDvecs[c, t, :] = TDvecs1[b_pos, :]
            # conv_TDvecs = conv_TDvecs.view(len(turn_nums), max(turn_nums), -1)
            # process data for modeling history turns with TDM
            turn_nums = [len(hist) for hist in one_data[2]]
            max_context_num = max([len(turn[1]) for turn in chain.from_iterable([h for h in one_data[2]])])
            TDM_batch_size = sum(turn_nums)
            hist_context = np.zeros((TDM_batch_size, max_context_num, vocab_size), dtype=np.int32)
            hist_targets = np.zeros((TDM_batch_size, vocab_size), dtype=np.int32)
            for u in range(len(one_data[2])):
                for t in range(len(one_data[2][u])):
                    b_pos = t + sum(turn_nums[:u])
                    hist_targets[b_pos, :] = one_data[2][u][t][0]
                    for i, row in enumerate(one_data[2][u][t][1]):
                        hist_context[b_pos][i][:] = row
            TDM_loss2, TDvecs2 = TDMmodel(hist_context, hist_targets, return_latent=True)
            hist_TDvecs = torch.zeros((len(turn_nums), max(turn_nums), TDvecs2.size(-1)))
            if torch.cuda.is_available():  # run in GPU
                hist_TDvecs = hist_TDvecs.cuda()
            for u in range(len(one_data[2])):
                for t in range(len(one_data[2][u])):
                    b_pos = t + sum(turn_nums[:u])
                    hist_TDvecs[u, t, :] = TDvecs2[b_pos, :]
            # history_TDvecs = history_TDvecs.view(len(turn_nums), max(turn_nums), -1)
            # produce predictions
            if train_style == 'onlyTDM':
                avg_loss += TDM_loss1.item() + TDM_loss2.item()
                loss = TDM_loss1 + TDM_loss2
            else:
                predictions = model(one_data[0], conv_TDvecs, hist_TDvecs)
                bce_loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
                if train_style == 'fixTDM':
                    avg_loss += bce_loss.item()
                    loss = bce_loss
                else:
                    avg_loss += bce_loss.item() + (TDM_loss1.item() + TDM_loss2.item()) * config.TDM_weight
                    loss = bce_loss + (TDM_loss1 + TDM_loss2) * config.TDM_weight
            # count += 1
            # if count % 1000 == 0:
            #     print('Epoch: %d, iterations: %d, loss: %g' % (epoch, count, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            if (train_style == 'joint' or train_style == 'fixTDM') and clip > 0:
                nn.utils.clip_grad_value_(model.parameters(), clip)
            if (train_style == 'joint' or train_style == 'onlyTDM') and clip > 0:
                nn.utils.clip_grad_value_(TDMmodel.parameters(), clip)
            optimizer.step()
        begin_idx = end_idx
    avg_loss /= (len(corp.convs_in_train)/config.batch_size)
    end = time.time()
    print('%s Train Epoch: %d done! Train avg_loss: %g!  Using time: %.2f minutes!' % (train_style, epoch, avg_loss, (end-start)/60))
    return avg_loss
