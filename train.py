import os
import sys
import random
import torch
import math
import time
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
from itertools import chain
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from corpus_process import Corpus, create_embedding_matrix, MyDataset
from joint_train import weighted_binary_cross_entropy, joint_train_epoch, joint_loss_compute, joint_evaluate
from TDMmodel import conv_models, criterions, encoders, model_bases, utils
from lstm_bia import LSTMBiA
from lstm_tdm import LSTMTDM
from conv_killer import ConvKiller


def evaluate(model, test_data, need_user_history=False, threshold=-1):  # evaluation metrics
    model.eval()
    true_labels = []
    pred_labels = []
    for step, one_data in enumerate(test_data):
        label = one_data[-1].data.numpy()
        if need_user_history:
            predictions = model(one_data[0], one_data[1])
        else:
            predictions = model(one_data[0])
        if torch.cuda.is_available():  # run in GPU
            pred_label = predictions.cpu().data.numpy()
        else:
            pred_label = predictions.data.numpy()
        true_labels = np.concatenate([true_labels, label])
        pred_labels = np.concatenate([pred_labels, pred_label])

    try:
        auc = roc_auc_score(true_labels, pred_labels)
    except ValueError:
        auc = 0.0
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


def loss_compute(model, using_data, loss_weights, need_user_history=False):
    model.eval()
    avg_loss = 0.0
    for step, one_data in enumerate(using_data):
        label = one_data[-1]
        if torch.cuda.is_available():  # run in GPU
            label = label.cuda()
        if need_user_history:
            predictions = model(one_data[0], one_data[1])
        else:
            predictions = model(one_data[0])
        loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
        avg_loss += loss.item()
    avg_loss /= len(using_data)
    return avg_loss


def train_epoch(model, corp, loss_weights, optimizer, epoch, config, need_user_history=False, clip=0):
    start = time.time()
    model.train()
    print('Epoch: %d start!' % epoch)
    avg_loss = 0.0
    count = 0
    begin_idx = 0
    random.seed(epoch)
    random.shuffle(corp.convs_in_train)
    random.seed(config.random_seed)
    train_len = len(corp.convs_in_train)
    while begin_idx < train_len:
        end_idx = begin_idx + config.batch_size * 100 if begin_idx + config.batch_size * 100 < train_len else train_len
        train_data = MyDataset(corp, corp.convs_in_train[begin_idx: end_idx], config.history_size, config.positive_sample_weight, use_BERT=False, use_TDM=False)
        sampler = data.WeightedRandomSampler(train_data.data_weight, num_samples=len(train_data), replacement=True)
        train_loader = data.DataLoader(train_data, collate_fn=train_data.collate_fn, batch_size=config.batch_size, num_workers=0, sampler=sampler)
        for step, one_data in enumerate(train_loader):
            # print len(one_data[0])
            label = one_data[-1]
            if torch.cuda.is_available():  # run in GPU
                label = label.cuda()
            if need_user_history:
                predictions = model(one_data[0], one_data[1])
            else:
                predictions = model(one_data[0])
            # print predictions, label
            loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
            avg_loss += loss.item()
            count += 1
            if count % 1000 == 0:
                print('Epoch: %d, iterations: %d, loss: %g' % (epoch, count, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_value_(model.parameters(), clip)
            optimizer.step()
        begin_idx = end_idx
    avg_loss /= (len(corp.convs_in_train) / config.batch_size)
    end = time.time()
    print('Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end - start) / 60))
    return avg_loss


def train(config):
    # Figure out the parameters
    filename = config.filename
    modelname = config.modelname
    need_history = (False if modelname == "ConvKiller" else True)
    if filename == "test":
        trainfile, testfile, validfile = "t_train.json", "t_test.json", "t_valid.json"
    elif filename == "twitter":
        trainfile, testfile, validfile = "twitter_train.json", "twitter_test.json", "twitter_valid.json"
    elif filename == "reddit":
        trainfile, testfile, validfile = "reddit_train.json", "reddit_test.json", "reddit_valid.json"
    else:
        print('Data name not correct!')
        sys.exit()
    use_TDM, use_BERT = False, False
    if modelname == 'LSTMTDM':
        use_TDM = True
    elif 'BERT' in modelname:
        use_BERT = True
    #print(use_BERT, use_TDM,'in train')
    corp = Corpus(trainfile, testfile, validfile, config.batch_size, config.history_size, use_BERT=use_BERT, use_TDM=use_TDM)
    #print(use_BERT, use_TDM, 'in train 1')
    #exit()
    config.vocab_num = corp.wordNum
    if config.no_pretrain_embedding or use_BERT or modelname == 'ConvKiller':
        config.embedding_matrix = None
    else:
        config.embedding_matrix = create_embedding_matrix(filename, corp.r_wordIDs, corp.wordNum, config.embedding_dim)
    res_path = "BestResults/" + modelname + "/" + filename + "/"
    mod_path = "BestModels/" + modelname + "/" + filename + "/"
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)

    # Set up the model
    if modelname == "LSTMBiA":
        model = LSTMBiA(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                        config.num_layer, config.model_num_layer, pretrained_weight=config.embedding_matrix)
        para_list = str(config.batch_size) + "_" + str(config.history_size) + "_" + str(config.lr) + "_" + \
            str(config.train_weight) + "_" + str(config.threshold) + "_" + str(config.runtime)
        mod_path += para_list + '.model'
        res_path += para_list + '.data'
    elif modelname == "ConvKiller":
        max_turn_num = max([len(corp.convs_ids[c]) for c in corp.convs_ids])
        max_turn_len = max([len(turn) for turn in chain.from_iterable([corp.convs_ids[c] for c in corp.convs_ids])]) - 1
        model = ConvKiller(config, max_turn_num, max_turn_len)
        para_list = str(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        mod_path += para_list + '.model'
        res_path += para_list + '.data'
    elif modelname == "LSTMTDM":
        model = LSTMTDM(config)
        TDMmodel = conv_models.TDM(corp, config)
        para_list = str(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        TDMmod_path = mod_path + para_list + '_TDM.model'
        f1_TDMmod_path = mod_path + para_list + '_TDM_f1.model'
        loss_TDMmod_path = mod_path + para_list + '_TDM_loss.model'
        f1_mod_path = mod_path + para_list + '_f1.model'
        loss_mod_path = mod_path + para_list + '_loss.model'
        res_path += para_list + '.data'
    else:
        print('Model name not correct!')
        sys.exit()
    loss_weights = torch.Tensor([1, config.train_weight])
    if torch.cuda.is_available():              # run in GPU
        model = model.cuda()
        loss_weights = loss_weights.cuda()
        if modelname == "LSTMTDM":
            TDMmodel = TDMmodel.cuda()
    if modelname == "LSTMTDM":
        # conv_lstm_params = list(map(id, model.conv_lstm.parameters()))
        # rest_params = filter(lambda p: id(p) not in conv_lstm_params, model.parameters())
        optimizer_LSTM = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_weight)
        optimizer_TDM = optim.Adam(TDMmodel.parameters(), lr=config.lr/2, weight_decay=config.l2_weight)
        # whole_params = chain(rest_params, TDMmodel.parameters())
        optimizer_whole = optim.Adam([{'params': model.parameters()}, {'params': TDMmodel.parameters(), 'lr': config.lr/2}], lr=config.lr, weight_decay=config.l2_weight)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_whole, T_max=10, eta_min=config.lr/2)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer_whole, lr_lambda=lambda epoch: 1.0 / ((epoch + 1) ** 0.5))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_weight)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / ((epoch + 1) ** 0.5))

    best_valid_auc = -1.0
    best_valid_loss = 9999999.9
    no_improve = 0
    # fix_no_improve = 0
    if modelname == "LSTMTDM":
        if config.use_pretrained_TDM is None:
            print('Only TDM train begin!')
            for epoch in range(config.TDM_train_epoch):
                p_loss = joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_TDM, epoch, config, 'onlyTDM', config.clip_value)
                if (epoch+1) % 10 == 0:
                    pretrain_path = 'PretrainedTDM/' + filename + '_epoch-' + str(epoch+1) + '_' + str(p_loss) + '_' + \
                                    TDMmod_path.split('/' + filename + '/')[1]
                    torch.save(TDMmodel.state_dict(), pretrain_path)
                    print('Pretrained TDM has been saved in: %s !' % pretrain_path)
        else:
            print('Load TDM model! Path: %s !' % config.use_pretrained_TDM)
            TDMmodel.load_state_dict(torch.load(config.use_pretrained_TDM))
        print('Fix TDM train begin!')
        for epoch in range(config.fixTDM_train_epoch):
            joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_LSTM, epoch, config, 'fixTDM', config.clip_value)
        print('Joint train begin!')
        # train_style = 'joint'
        for epoch in range(config.max_epoch):
            train_loss = joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_whole, epoch, config, 'joint', config.clip_value)
            if math.isnan(train_loss):
                break
            # if train_style == 'joint':
            #     joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_whole, epoch, config, train_style, config.clip_value)
            # else:
            #     joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_LSTM, epoch, config, train_style, config.clip_value)
            valid_auc, valid_loss = joint_evaluate(model, TDMmodel, corp.valid_loader, config.threshold, loss_weights)
            # valid_loss = joint_loss_compute(model, TDMmodel, corp.valid_loader, loss_weights, config.TDM_weight)
            # if best_valid_loss is None or valid_loss < best_valid_loss:
            if best_valid_auc < valid_auc or best_valid_loss > valid_loss:
                no_improve = 0
                # fix_no_improve = 0
                if best_valid_auc < valid_auc:
                    best_valid_auc = valid_auc
                    best_epoch = epoch
                    os.system('rm ' + f1_mod_path)
                    os.system('rm ' + f1_TDMmod_path)
                    print('New Best Valid AUC Result!!! Valid AUC: %g' % best_valid_auc)
                    torch.save(model.state_dict(), f1_mod_path)
                    torch.save(TDMmodel.state_dict(), f1_TDMmod_path)
                else:
                    print('Best Valid AUC: %g, Current Valid AUC: %g' % (best_valid_auc, valid_auc))
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_loss_epoch = epoch
                    os.system('rm ' + loss_mod_path)
                    os.system('rm ' + loss_TDMmod_path)
                    print('New Best Valid Loss Result!!! Valid Loss: %g' % best_valid_loss)
                    torch.save(model.state_dict(), loss_mod_path)
                    torch.save(TDMmodel.state_dict(), loss_TDMmod_path)
                else:
                    print('Best Valid Loss: %g, Current Valid Loss: %g' % (best_valid_loss, valid_loss))
            else:
                no_improve += 1
                print('Best Valid AUC: %g, Current Valid AUC: %g' % (best_valid_auc, valid_auc))
                print('Best Valid Loss: %g, Current Valid Loss: %g' % (best_valid_loss, valid_loss))
                if no_improve == 10:
                    break
                # if train_style == 'fixTDM':
                #     fix_no_improve += 1
                if no_improve == 5:
                    joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_TDM, epoch, config, 'onlyTDM', config.clip_value)
                    joint_train_epoch(model, TDMmodel, corp, loss_weights, optimizer_TDM, epoch, config, 'fixTDM', config.clip_value)
                    # fix_no_improve = 0
                # train_style = ('fixTDM' if train_style == 'joint' or fix_no_improve < 2 else 'joint')
            scheduler.step()
    else:
        for epoch in range(config.max_epoch):
            train_epoch(model, corp, loss_weights, optimizer, epoch, config, need_history, config.clip_value)
            _, _, _, _, valid_auc, _ = evaluate(model, corp.valid_loader, need_history, config.threshold)
            # valid_loss = loss_compute(model, corp.valid_loader, loss_weights, need_history)
            # if best_valid_loss is None or valid_loss < best_valid_loss:
            if best_valid_auc < valid_auc:
                no_improve = 0
                best_valid_auc = valid_auc
                best_epoch = epoch
                os.system('rm ' + mod_path)
                print('New Best Valid Result!!! Valid AUC: %g' % best_valid_auc)
                torch.save(model.state_dict(), mod_path)
            else:
                no_improve += 1
                print('Best Valid AUC: %g, Current Valid AUC: %g' % (best_valid_auc, valid_auc))
            if no_improve == 8:
                break
            scheduler.step()
    if 'TDM' not in modelname:
        model.load_state_dict(torch.load(mod_path))
        res = evaluate(model, corp.test_loader, need_history, config.threshold)
    else:
        model.load_state_dict(torch.load(f1_mod_path))
        TDMmodel.load_state_dict(torch.load(f1_TDMmod_path))
        res_auc = joint_evaluate(model, TDMmodel, corp.test_loader, config.threshold)
        model.load_state_dict(torch.load(loss_mod_path))
        TDMmodel.load_state_dict(torch.load(loss_TDMmod_path))
        res_loss = joint_evaluate(model, TDMmodel, corp.test_loader, config.threshold)
        if res_auc[1] >= res_loss[1]:
            res = res_auc
            print("Best results achieve by best valid AUC!")
            os.system('rm ' + loss_mod_path)
            os.system('rm ' + loss_TDMmod_path)
        else:
            res = res_loss
            print("Best results achieve by best valid loss!")
            os.system('rm ' + f1_mod_path)
            os.system('rm ' + f1_TDMmod_path)
    print('Result in test set: Accuracy %g, F1 Score %g, Precision %g, Recall %g, AUC %g' % (res[0], res[1], res[2], res[3], res[4]))
    with open(res_path, 'w') as f:
        f.write('Accuracy\tF1-Score\tPrecision\tRecall\tAUC\n')
        f.write('%g\t%g\t%g\t%g\t%g\n\n' % (res[0], res[1], res[2], res[3], res[4]))
        f.write('Threshold: %g\n' % res[5])
        if modelname == "LSTMTDM" and res == res_loss:
            f.write('Best Valid Loss: %g\n' % best_valid_loss)
            f.write('Best epoch: %d\n' % best_loss_epoch)
        else:
            f.write('Best Valid AUC: %g\n' % best_valid_auc)
            f.write('Best epoch: %d\n' % best_epoch)
        f.write('\nParameters:\n')
        for key in config.__dict__:
            f.write('%s : %s\n' % (key, config.__dict__[key]))


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, choices=["test", "twitter", "reddit"])
    parser.add_argument("modelname", type=str, choices=["LSTMOri", "LSTMBiA", "LSTMTDM", "ConvKiller"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--history_size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip_value", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--model_num_layer", type=int, default=2)
    parser.add_argument("--no_pretrain_embedding", action="store_true")
    parser.add_argument("--runtime", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=-1)
    parser.add_argument("--train_weight", type=float, default=1)
    parser.add_argument("--positive_sample_weight", type=float, default=1)
    parser.add_argument("--TDM_weight", type=float, default=1)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--d_size", type=int, default=1)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--k_size", type=int, default=20)
    parser.add_argument('--no_topic_init', dest='topic_init', action='store_false')
    parser.add_argument('--no_topic_concat', dest='topic_concat', action='store_false')
    parser.add_argument('--no_disc_lstm', dest='disc_lstm', action='store_false')
    parser.add_argument('--no_dd_att', dest='dd_att', action='store_false')
    parser.add_argument('--use_l1_reg', dest='use_l1_reg', action='store_true')
    parser.add_argument("--no_use_gpu", dest='use_gpu', action='store_false')
    parser.add_argument("--TDM_train_epoch", type=int, default=60)
    parser.add_argument("--fixTDM_train_epoch", type=int, default=5)
    parser.add_argument("--use_pretrained_TDM", type=str, default=None)  # if use, clarify the model path here
    parser.add_argument("--model_type", type=float, default=1.1)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--use_CNN", dest='use_CNN', action='store_true')
    parser.add_argument("--no_use_LSTM", dest='use_LSTM', action='store_false')
    parser.add_argument("--CNN_kernal_num", type=int, default=100)
    parser.add_argument("--l2_weight", type=float, default=0)

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    config = parse_config()
    setup_seed(config.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    train(config)


