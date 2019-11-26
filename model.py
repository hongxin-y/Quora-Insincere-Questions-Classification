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
        self.cated_dense = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p = 0.5)
        self.batchnorm = nn.BatchNorm1d(128)

        self.final_dense = nn.Linear(128, 2)
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
        # spatial dropout(0.5)
        #seq = seq.permute(0,2,1)
        #seq = F.dropout2d(seq, p = 0.5, training=True, inplace=False)
        #seq = seq.permute(0,2,1)

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
        context = F.relu(context, inplace = True)

        # stats features embedding(to batchsize*64)
        stat_features = self.stat_dense(stat_features)
        stat_features = F.relu(stat_features, inplace = True)
        #stat_features = self.dropout(stat_features)

        # concat two embeddings(to batchsize*128)
        output = torch.cat((context, stat_features), 1)
        # full connection layer with dropout(0.5) and batchnorm layer(to batchsize*128)
        output = self.cated_dense(output)
        output = self.batchnorm(output)
        output = F.relu(output, inplace = True)
        #output = self.dropout(output)
        

        # final output, 2 classes(to batchsize*2)
        output = self.final_dense(output)
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
    #stats_features = (stats_features - torch.mean(stats_features, dim = 0))/torch.std(stats_features, dim = 0)
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
        batch_X = list(map(lambda sentence: torch.tensor(list(map(lambda index: word2vec[index2word[index]] if index2word[index] in word2vec.keys() else np.zeros(vec_dim), sentence))), batch_sentences))
        '''
        for sentence in batch_sentences:
            vec = []
            for index in sentence:
                if index2word[index] not in word2vec.keys():
                    vec.append(np.zeros(vec_dim))
                else:
                    vec.append(word2vec[index2word[index]])
            batch_X.append(torch.tensor(vec))
        '''
        batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
        batch_F = stats_features[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)

        logit = model.forward(batch_X, batch_L, batch_F)
        prob = F.softmax(logit, dim = 1)[:,1]

        res = np.where(prob.cpu() > threshold, 1, 0)
        ret = np.r_[ret, res]
    return ret

def evaluate(model, valid_sentences, valid_length, valid_stats_features, valid_targets, batch_size = 100, threshold = 0.5):
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
    print("AUC: {}".format(metrics.auc(fpr, tpr)))
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

def weighted_mse_loss(input, target, class_weights):
    class_weights = np.array(class_weights)
    weights = torch.tensor(np.where(target == 1, class_weights[1], class_weights[0]))
    return torch.sum(weights * (input - target) ** 2)

def train(model, train_sentences, train_length, train_stats_features, train_targets, index2word, word2vec, learning_rate = 0.001, batch_size = 100, optimizer = "Adam", iterations = 100, threshold = 0.5, training = True):
    vec_dim = word2vec['how'].shape[0]
    
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
                batch_X = list(map(lambda sentence: torch.tensor(list(map(lambda index: word2vec[index2word[index]] if index2word[index] in word2vec.keys() else np.zeros(vec_dim), sentence))), batch_sentences))
                '''
                for sentence in batch_sentences:
                    vec = []
                    for index in sentence:
                        if index2word[index] not in word2vec.keys():
                            vec.append(np.zeros(vec_dim))
                        else:
                            vec.append(word2vec[index2word[index]])
                    batch_X.append(torch.tensor(vec))
                '''
                batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
                batch_Y = train_targets[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)
                batch_F = train_stats_features[k*batch_size:(k+1)*batch_size][idx].to(DEVICE)

                logit = model.forward(batch_X, batch_L, batch_F)
                probs = F.softmax(logit, dim = 1)
                #pos_ratio = (batch_Y.long() == 1).sum().float()/batch_Y.long().shape[0]
                #neg_ratio = (batch_Y.long() == 0).sum().float()/batch_Y.long().shape[0]
                loss = dice_loss(probs, torch.cat((batch_Y[:,None], 1-batch_Y[:,None]), dim = 1))
                loss += 0.5*F.cross_entropy(logit, batch_Y.long())#, weight = torch.tensor([neg_ratio, pos_ratio]).to(DEVICE), size_average=True)
                #loss += weighted_mse_loss(probs, batch_Y, [neg_ratio, pos_ratio])
                loss.backward()
                optimizer.step()
                e = time.perf_counter()
                #print("time cost this batch: {}".format(e-s), end = "")
                print("loss in epoch {} and batch {}: {}".format(epoch+1, k+1, loss))
                losses.append(float(loss))
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

def bootstrap_select(sentence2index_1, sentence2index_0, length_1, length_0, targets_1, targets_0, statistics_feature_1, statistics_feature_0, sample_num = 10000, pos_sample_rate = 0.5):
    num_pos = int(sample_num*pos_sample_rate)
    idx_pos = np.random.choice(range(len(sentence2index_1)), size = num_pos, replace = True)
    idx_neg = np.random.choice(range(len(sentence2index_0)), size = sample_num - num_pos, replace = True)
    sentence2index = np.r_[np.array(sentence2index_1)[idx_pos], np.array(sentence2index_0)[idx_neg]]
    target = np.r_[np.array(targets_1)[idx_pos], np.array(targets_0)[idx_neg]]
    statistics_feature = np.r_[np.array(statistics_feature_1)[idx_pos], np.array(statistics_feature_0)[idx_neg]]
    length = np.r_[np.array(length_1)[idx_pos], np.array(length_0)[idx_neg]]
    shuffle_idx = np.random.permutation(length.shape[0])
    return sentence2index[shuffle_idx], torch.tensor(length[shuffle_idx]), torch.tensor(target[shuffle_idx]), torch.tensor(statistics_feature[shuffle_idx])

if __name__ == "__main__":
    VALID_SAMPLE_RATE = 0.2

    sentence2index_1, sentence2index_0, targets_1, targets_0, \
           statistics_feature_1, statistics_feature_0, index2word, word2vec = getData()

    # Cross-validation method split
    train_sentences_1, train_length_1, train_stats_features_1, train_targets_1, \
        valid_sentences_1, valid_length_1, valid_stats_features_1, valid_targets_1 = \
        split_train_valid(sentence2index_1, statistics_feature_1, targets_1, valid_sample_rate = VALID_SAMPLE_RATE)
    train_sentences_0, train_length_0, train_stats_features_0, train_targets_0, \
        valid_sentences_0, valid_length_0, valid_stats_features_0, valid_targets_0 = \
        split_train_valid(sentence2index_0, statistics_feature_0, targets_0, valid_sample_rate = VALID_SAMPLE_RATE)

    # resample samples to balance data
    # positive sample is toxic questions here
    train_sentences, train_length, train_targets, train_stats_features = bootstrap_select(train_sentences_1, train_sentences_0, train_length_1, train_length_0, train_targets_1, train_targets_0, \
        train_stats_features_1, train_stats_features_0, sample_num = 800000, pos_sample_rate = 0.25)
    
    # concat validation set
    valid_sentences = np.r_[np.array(valid_sentences_1), np.array(valid_sentences_0)]
    valid_length = torch.tensor(np.r_[np.array(valid_length_1), np.array(valid_length_0)])
    valid_stats_features = torch.tensor(np.r_[np.array(valid_stats_features_1), np.array(valid_stats_features_0)])
    valid_targets = torch.tensor(np.r_[np.array(valid_targets_1), np.array(valid_targets_0)])
    
    # normalize stats features
    train_stats_mean = torch.mean(train_stats_features, dim=0)
    train_stats_std = torch.std(train_stats_features, dim=0)
    train_stats_features = (train_stats_features - train_stats_mean)/train_stats_std
    valid_stats_features = (valid_stats_features - train_stats_mean)/train_stats_std

    # delete useless tmp objects
    del sentence2index_1, sentence2index_0, targets_1, targets_0, statistics_feature_1, statistics_feature_0
    del train_sentences_0, train_sentences_1, train_length_0, train_length_1, train_stats_features_0, train_stats_features_1, train_targets_0, train_targets_1
    del valid_sentences_0, valid_sentences_1, valid_length_0, valid_length_1, valid_stats_features_0, valid_stats_features_1, valid_targets_0, valid_targets_1

    for th in [0.5]:
        #with open("./model", 'rb') as f:
        model = quora_detector(word2vec['how'].shape[0], len(train_stats_features[0]))
        train(model, train_sentences, train_length, train_stats_features, train_targets, index2word, word2vec, \
            learning_rate = 0.00001, batch_size = 512, optimizer = "Adam", iterations = 1, threshold = th, training=True)
        
        # evaluation on training set
        print("evaluation on training set")
        evaluate(model, train_sentences, train_length, train_stats_features, train_targets, batch_size = 512, threshold = th)
        print("evaluation on validation set")
        # evaluation on validation set
        evaluate(model, valid_sentences, valid_length, valid_stats_features, valid_targets, batch_size = 512, threshold = th)

    for th in np.arange(0, 1, 0.05):
        with open("./model", 'rb') as f:
            model = pickle.load(f)
            # evaluation on training set
            print("evaluation on training set")
            evaluate(model, train_sentences, train_length, train_stats_features, train_targets, batch_size = 512, threshold = th)
            print("evaluation on validation set")
            # evaluation on validation set
            evaluate(model, valid_sentences, valid_length, valid_stats_features, valid_targets, batch_size = 512, threshold = th)