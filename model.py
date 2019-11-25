import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import os
import pickle
import time

from utils import preprocess
from utils import getData

torch.manual_seed(123)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quora_detector(nn.Module):
    def __init__(self, vec_dim, fea_num):
        super(quora_detector, self).__init__()
        self.rnn = nn.LSTM(input_size = vec_dim, hidden_size = 64, batch_first = True, bidirectional = True)
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.stat_dense = nn.Linear(fea_num, 64)
        self.final_dense = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p = 0.5)
        self.batchnorm = nn.BatchNorm1d(2)
        self.initialize()

    # initialize with Xavier distribution
    def initialize(self):
        w = self.rnn.all_weights
        nn.init.xavier_uniform_(w[0][0])
        nn.init.xavier_uniform_(w[0][1])
        nn.init.xavier_uniform_(w[1][0])
        nn.init.xavier_uniform_(w[1][1])

    def last_element(self, output, length):
        ret = []
        for i in range(output.shape[0]):
            ret.append(output[:,int(length[i]-1),:][i])
        return torch.cat(ret,0).view(output.shape[0],-1)

    def forward(self, seq, length, stat_features):
        # context embedding
        # sparsy dropout(0.5)
        seq = seq.permute(0,2,1)
        seq = F.dropout2d(seq, p = 0.5, training=True, inplace=False)
        seq = seq.permute(0,2,1)
        # unpadding and RNN(128) training
        seq = torch.nn.utils.rnn.pack_padded_sequence(seq, length, batch_first = True)
        context, _ = self.rnn(seq)
        context, _ = torch.nn.utils.rnn.pad_packed_sequence(context, batch_first = True)
        # take the last layer output(to batchsize*1*128)
        context = self.last_element(context, length)[:,:,None]
        context = context.permute(0,2,1)
        # 1d convolution layer, 64 convolution kernels(to batchsize*64*128)
        context = self.conv(context)
        # global max pooling(to batchsize*64(*1))
        context = F.max_pool1d(context, kernel_size = context.shape[2:])
        context = context[:,:,-1]

        # stats features embedding(to batchsize*64)
        stat_features = self.stat_dense(stat_features)
        stat_features = F.relu(stat_features, inplace = True)

        # concat two embeddings(to batchsize*128)
        output = torch.cat((context, stat_features), 1)
        # full connection layer with dropout(0.5) and batchnorm layer(to batchsize*2)
        output = self.final_dense(output)
        output = self.dropout(output)
        output = self.batchnorm(output)
        return output

'''
# random generated test case
x = torch.randn(32,35,300)
y = torch.randn(32,10)
l = torch.ones(x.shape[0])*x.shape[1]
x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).float()
model = quora_detector(300, 10)
model.forward(x, l, y)
'''

def split_train_valid(sentences, stats_features, targets, valid_sample_rate = 0.1):
    length = torch.tensor(stats_features)[:,-1]
    stats_features = torch.tensor(stats_features).float()
    targets = torch.tensor(targets).float()
        
    shuffle_idx = np.random.permutation(length.shape[0])
    valid_set_len = int(valid_sample_rate*length.shape[0])

    valid_length = length[shuffle_idx[:valid_set_len]]
    train_length = length[shuffle_idx[valid_set_len:]]
    valid_stats_features = stats_features[shuffle_idx[:valid_set_len]]
    train_stats_features = stats_features[shuffle_idx[valid_set_len:]]
    valid_targets = targets[shuffle_idx[:valid_set_len]]
    train_targets = targets[shuffle_idx[valid_set_len:]]
    valid_sentences = np.array(sentences)[shuffle_idx[:valid_set_len]]
    train_sentences = np.array(sentences)[shuffle_idx[valid_set_len:]]
    return train_sentences, train_length, train_stats_features, train_targets, valid_sentences, valid_length, valid_stats_features, valid_targets

def predict(model, sentences, length, stats_features, batch_size = 100, threshold = 0.5):
    vec_dim = word2vec['how'].shape[0]
    ret = np.array([])
    for k in range((len(sentences)-1)//batch_size + 1):
        batch_L = length[k*batch_size:(k+1)*batch_size]
        batch_L, idx = torch.sort(batch_L, descending = True)
        batch_L = batch_L.to(DEVICE)
        batch_sentences = sentences[k*batch_size:(k+1)*batch_size]
        batch_X = []
        for sentence in batch_sentences:
            vec = []
            for index in sentence:
                if index2word[index] not in word2vec.keys():
                    vec.append(np.zeros(vec_dim))
                else:
                    vec.append(word2vec[index2word[index]])
            batch_X.append(torch.tensor(vec))
        batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
        batch_F = stats_features[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)

        logit = model.forward(batch_X, batch_L, batch_F)
        prob = F.softmax(logit, dim = 1)[:,1]

        res = np.where(prob.cpu() > threshold, 1, 0)
        ret = np.r_[ret, res]
    return ret

def evalutate(model, valid_sentences, valid_length, valid_stats_features, valid_targets, batch_size = 100, threshold = 0.5):
    model.eval()
    # threshold is for toxic sentences, i.e. model will judge a sentence toxic when output_sofrmax(toxic) > threshold
    eval_start = time.perf_counter()
    valid_pred = predict(model, valid_sentences, valid_length, valid_stats_features, batch_size = batch_size, threshold = threshold)
    acc = accuracy_score(valid_targets, valid_pred)
    f1 = f1_score(valid_targets, valid_pred)
    confusion = confusion_matrix(valid_targets, valid_pred)
    eval_end = time.perf_counter()
    print("Total accuracy computation time: {}".format(eval_end - eval_start))
    print("Accuracy on validation set:{}\nF1 score of validation set:{}".format(acc, f1))
    print("Confusion Matrix: \n", confusion)
    fpr, tpr, _ = metrics.roc_curve(valid_targets, valid_pred)
    print("AUC: {}", metrics.auc(fpr, tpr))
    print("Accuracy on ground truth 0(TNR): {}".format(1-fpr[1]))
    print("Accuracy on ground truth 1(TPR,recall): {}".format(tpr[1]))
    print("Balanced accuracy: {}".format((1-fpr[1] + tpr[1])/2))

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def train_eval(model, sentences, stats_features, targets, index2word, word2vec, valid_sample_rate = 0.1, learning_rate = 0.001, batch_size = 100, optimizer = "Adam", iterations = 100, threshold = 0.5, training = True):
    vec_dim = word2vec['how'].shape[0]
    
    # Cross-validation method split with rate
    train_sentences, train_length, train_stats_features, train_targets, \
        valid_sentences, valid_length, valid_stats_features, valid_targets = \
        split_train_valid(sentences, stats_features, targets, valid_sample_rate = valid_sample_rate)

    if training:
        if optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        # put model on GPU and parallelize
        model.to(DEVICE)
        model = nn.DataParallel(model)

        print("Training will on GPU" if DEVICE == torch.device("cuda:0") else "Training will on CPU")

        losses = []
        #train_errors = []
        #test_errors = []

        model.train()
        train_start_time = time.perf_counter()
        for epoch in range(iterations):
            for k in range((len(train_sentences)-1)//batch_size + 1):
                s = time.perf_counter()
                # free unreferenced cuda memory and put gradient zero
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                batch_L = train_length[k*batch_size:(k+1)*batch_size]
                batch_L, idx = torch.sort(batch_L, descending = True)
                batch_L = batch_L.to(DEVICE)
                batch_sentences = train_sentences[k*batch_size:(k+1)*batch_size]
                batch_X = []
                for sentence in batch_sentences:
                    vec = []
                    for index in sentence:
                        if index2word[index] not in word2vec.keys():
                            vec.append(np.zeros(vec_dim))
                        else:
                            vec.append(word2vec[index2word[index]])
                    batch_X.append(torch.tensor(vec))
                batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
                batch_Y = train_targets[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)
                batch_F = train_stats_features[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)

                logit = model.forward(batch_X, batch_L, batch_F)
                res = torch.round(F.softmax(logit, dim = 1)[:,1] / threshold * 0.5)
                #pos_neg_ratio = (batch_Y.long() == 1).sum().float()/(batch_Y.long() == 0).sum()
                loss = dice_loss(res, batch_Y)#F.cross_entropy(logit, batch_Y.long())#, weight = torch.tensor([1, pos_neg_ratio]).to(DEVICE), size_average=True)
                
                loss.backward()
                optimizer.step()
                e = time.perf_counter()
                #print("time cost this batch: {}".format(e-s), end = "")
                print("loss in epoch {} and batch {}: {}".format(epoch+1, k+1, loss))
                #losses.append(float(loss))
                #valid_pred = predict(model, valid_sentences, valid_length, valid_stats_features, batch_size = batch_size, threshold = threshold)
                #train_pred = predict(model, train_sentences, train_length, train_stats_features, batch_size = batch_size, threshold = threshold)
                #test_errors.append(float(accuracy_score(valid_targets, valid_pred)))
                #train_errors.append(float(accuracy_score(train_targets, train_pred)))
            print("last loss in epoch {}: {}".format(epoch+1, loss))
        train_stop_time = time.perf_counter()
        print("Total Training Time: {} seconds".format(train_stop_time - train_start_time))

        # save model into binary file
        with open("./model", 'wb') as f:
            pickle.dump(model, f)

        # show and save loss-iteration plot
        plt.plot(range(1, 1 + len(losses)), losses)
        plt.xlabel("iterations")
        plt.ylabel("loss function")
        plt.savefig("loss.jpg")
        #plt.show()
        '''
        plt.plot(range(1, 1 + len(test_errors)), test_errors)
        plt.show()
        plt.plot(range(1, 1 + len(train_errors)), train_errors)
        plt.show()
        '''

        # evaluation on validation set
        evalutate(model, valid_sentences, valid_length, valid_stats_features, valid_targets, batch_size = batch_size, threshold = threshold)

def bootstrap_select(sentence2index_1, sentence2index_0, targets_1, targets_0, statistics_feature_1, statistics_feature_0, sample_num = 10000, pos_sample_rate = 0.5):
    num_pos = int(sample_num*pos_sample_rate)
    idx_pos = np.random.choice(range(len(sentence2index_1)), size = num_pos, replace = True)
    idx_neg = np.random.choice(range(len(sentence2index_0)), size = sample_num - num_pos, replace = True)
    sentence2index = np.r_[np.array(sentence2index_1)[idx_pos], np.array(sentence2index_0)[idx_neg]]
    target = np.r_[np.array(targets_1)[idx_pos], np.array(targets_0)[idx_neg]]
    statistics_feature = np.r_[np.array(statistics_feature_1)[idx_pos], np.array(statistics_feature_0)[idx_neg]]
    return sentence2index, target, statistics_feature

if __name__ == "__main__":
    sentence2index_1, sentence2index_0, targets_1, targets_0, \
           statistics_feature_1, statistics_feature_0, index2word, word2vec = getData()
    # positive sample is toxic questions here
    sentence2index, train_targets, train_stats_features = bootstrap_select(sentence2index_1, sentence2index_0, targets_1, targets_0, \
        statistics_feature_1, statistics_feature_0, sample_num = 1000000, pos_sample_rate = 0.3)
    del sentence2index_1, sentence2index_0, targets_1, targets_0, statistics_feature_1, statistics_feature_0
    for th in [0.5]:
        #with open("./model", 'rb') as f:
        model = quora_detector(word2vec['how'].shape[0], len(train_stats_features[0]))
        train_eval(model, sentence2index, train_stats_features, train_targets, index2word, word2vec, \
            valid_sample_rate = 0.2, learning_rate = 0.001, batch_size = 512, optimizer = "Adam", iterations = 35, threshold = th, training=True)