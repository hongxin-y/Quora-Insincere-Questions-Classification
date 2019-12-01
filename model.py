import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from tqdm import tqdm
import os
import pickle
import time

torch.manual_seed(123)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quora_detector(nn.Module):
    def __init__(self, vec_dim, fea_num=5, batch_size=512):
        super(quora_detector, self).__init__()
        self.batch_size = batch_size
        self.rnn_1 = nn.LSTM(input_size=vec_dim, hidden_size=64, batch_first=True, bidirectional=True)
        self.rnn_2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # self.maxPool = torch.max

        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        self.stats_dense = nn.Linear(fea_num, 64)

        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 64)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, 1)

        self.initialize()

    # initialize with Xavier distribution
    def initialize(self):
        w = self.rnn_1.all_weights
        nn.init.xavier_uniform_(w[0][0])
        nn.init.xavier_uniform_(w[0][1])
        nn.init.xavier_uniform_(w[1][0])
        nn.init.xavier_uniform_(w[1][1])
        w = self.rnn_2.all_weights
        nn.init.xavier_uniform_(w[0][0])
        nn.init.xavier_uniform_(w[0][1])
        nn.init.xavier_uniform_(w[1][0])
        nn.init.xavier_uniform_(w[1][1])

    def forward(self, seq, length, statistics):
        # spatial dropout(0.1)
        seq = seq.permute(0,2,1)
        seq = F.dropout2d(seq, p = 0.1, training=True, inplace=False)
        seq = seq.permute(0,2,1)

        context = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True)
        context, _ = self.rnn_1(context)
        context, _ = self.rnn_2(context)
        context, _ = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)
        # context, _ = self.maxPool(context, 1)
        context = self.last_element(context, length)[:,:,None]
        context = context.permute(0,2,1)
        # 1d convolution layer, 64 convolution kernels(to batchsize*64*128)
        context = self.conv(context)
        # global max pooling(to batchsize*64(*1))
        context = F.max_pool1d(context, kernel_size = context.shape[2:]).squeeze()

        statistics = self.stats_dense(statistics)
        context = torch.cat((context, statistics), 1)
        context = self.fc1(context)
        context = self.relu1(context)
        context = self.bn1(context)
        context = self.fc2(context)
        context = self.relu2(context)
        context = self.bn2(context)

        output = self.out(context)
        return output

    def last_element(self, output, length):
        ret = []
        for i in range(output.shape[0]):
            ret.append(output[:,int(length[i]-1),:][i])
        return torch.cat(ret,0).view(output.shape[0],-1)

'''
# random generated test case
x = torch.randn(32,35,300)
y = torch.randn(32,10)
l = torch.ones(x.shape[0])*x.shape[1]
x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).float()
model = quora_detector(300, 10)
model.forward(x, l, y)
'''


# Convert values to embeddings
def text_to_array(text):
    empyt_emb = np.zeros(300)
    text = text[:-1].split()
    embeds = []
    for x in text:
        if x in embeddings_index:
            embeds.append(embeddings_index[x])
        else:
            embeds.append(np.zeros(300))
    # embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds += [empyt_emb]  # * (30 - len(embeds))
    return np.array(embeds)

def text_to_statistics(text):
    stat_feature = []
    special_words = ["stupid", "discriminat", "dictatorship", "black", "gay", "race", "penis", "hypocritical", "kill",
                     "muslim", "sex", "political", "fuck", "dick", "vagina"]
    length_text = len(text)
    n_words = len(text.split(" "))
    stat_feature.append(len(text)/100)
    stat_feature.append(n_words / 50)
    cap_count = 0
    punc_count = 0
    for c in text:
        if c >= "A" and c <= "Z":
            cap_count += 1
        if c != " " and (c < "0" or (c > "9" and c < "A") or (c > "Z" and c < "a") or c > "z"):
            punc_count += 1
    stat_feature.append(cap_count / length_text)
    stat_feature.append(punc_count / length_text)
    sentence = text.lower()
    special_words_count = 0
    for special_word in special_words:
        special_words_count += sentence.count(special_word)
    stat_feature.append(special_words_count / n_words)
    return stat_feature


def predict(model, sentences, batch_size=100, threshold=0.5, pred_bar=True):
    sentences = np.array(sentences)
    length = torch.tensor(list(map(word_len, sentences)))
    ret = np.array([])
    iterator = tqdm(range((len(sentences) - 1) // batch_size + 1)) if pred_bar == True else range(
        (len(sentences) - 1) // batch_size + 1)
    for k in iterator:
        batch_L = length[k * batch_size:(k + 1) * batch_size]
        batch_L, idx = torch.sort(batch_L, descending=True)
        batch_L = batch_L.to(DEVICE)

        batch_sentences = sentences[k * batch_size:(k + 1) * batch_size][idx]

        batch_X = [torch.tensor(text_to_array(sentence)) for sentence in batch_sentences]
        batch_X_stats = [text_to_statistics(sentence) for sentence in batch_sentences]

        batch_X = nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
        batch_X_stats = torch.tensor(batch_X_stats).float().to(DEVICE)
        logit = model.forward(batch_X, batch_L, batch_X_stats)[:, 0]

        prob = torch.sigmoid(logit)

        # recover the order the data
        _, rev_idx = torch.sort(idx)
        res = torch.tensor(np.where(prob.cpu() > threshold, 1, 0))[rev_idx]
        ret = np.r_[ret, res]
    return ret


def word_len(sentence):
    return len(sentence.split())


def train(model, train_df, val_df, learning_rate=0.001, batch_size=100, optimizer="Adam", iterations=100, threshold=0.5,
          step_per_epoch=1000):
    torch.manual_seed(123)
    train_sentences = np.array(train_df['question_text'])
    train_targets = torch.tensor(np.array(train_df['target']))
    train_length = torch.tensor(list(map(word_len, train_sentences)))

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # put model on GPU
    model.to(DEVICE)
    # model = nn.DataParallel(model)

    print("Training will on GPU" if DEVICE == torch.device("cuda:0") else "Training will on CPU")

    losses = []
    # train_errors = []
    # test_errors = []

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    train_start_time = time.perf_counter()
    for epoch in range(iterations):
        shuffle_idx = np.random.permutation(train_sentences.shape[0])
        train_sentences = train_sentences[shuffle_idx]
        train_targets = train_targets[shuffle_idx]
        train_length = train_length[shuffle_idx]
        step = min(step_per_epoch, (len(train_sentences) - 1) // batch_size + 1)
        with tqdm(total=step, desc='epoch {}'.format(epoch + 1)) as t:
            for k in range(step):
                # free unreferenced cuda memory and put gradient zero
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                batch_L = train_length[k * batch_size:(k + 1) * batch_size]
                batch_L, idx = torch.sort(batch_L, descending=True)
                batch_L = batch_L.to(DEVICE)

                batch_sentences = train_sentences[k * batch_size:(k + 1) * batch_size][idx]

                batch_X = [torch.tensor(text_to_array(sentence)) for sentence in batch_sentences]
                batch_X_stats = [text_to_statistics(sentence) for sentence in batch_sentences]

                batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
                batch_X_stats = torch.tensor(batch_X_stats).float().to(DEVICE)
                batch_Y = train_targets[k * batch_size:(k + 1) * batch_size][idx].float().to(DEVICE)

                logit = model.forward(batch_X, batch_L, batch_X_stats)[:, 0]
                loss = criterion(logit, batch_Y)

                loss.backward()
                optimizer.step()

                losses.append(float(loss))
                t.set_postfix(loss='{:^7.3f}'.format(loss))
                t.update()
        # print("last loss in epoch {}: {}".format(epoch+1, loss))
    train_stop_time = time.perf_counter()
    print("Total Training Time: {} seconds".format(train_stop_time - train_start_time))

    # show and save loss-iteration plot
    plt.plot(range(1, 1 + len(losses)), losses)
    plt.xlabel("iterations")
    plt.ylabel("loss function")
    plt.savefig("loss.jpg")

def evaluation(model, data_df):
    prediction = predict(model, data_df['question_text'], batch_size = 512)
    acc = accuracy_score(data_df['target'], prediction)
    f1 = f1_score(data_df['target'], prediction)
    fpr, tpr, _ = metrics.roc_curve(data_df['target'], prediction)
    log = "ACC: "+str(acc)+"\nAUC: "+str(metrics.auc(fpr, tpr))+"\nf1: "+str(f1)+\
        "\nrecall: "+str(recall_score(data_df['target'], prediction))+"\nprecision: "+str(precision_score(data_df['target'],prediction))+\
            "\nBalanced Acc: " + str((1-fpr[1] + tpr[1])/2) + "\nTNR: " + str(1-fpr[1]) + "\n"
    return log, f1

def resample(train_df, pos_neg_ratio):
    new_neg_df = train_df[(train_df['target'] == 0)]
    num_pos = int(pos_neg_ratio*new_neg_df.shape[0])
    new_pos_df = train_df[(train_df['target'] == 1)].sample(num_pos, replace = True)
    train_df = pd.concat([new_neg_df, new_pos_df]).sample(frac = 1)
    return train_df

if __name__ == "__main__":
    VALID_SAMPLE_RATE = 0.1

    embeddings_index = KeyedVectors.load("./embeddings/glove.840B.300d.gensim")
    print('finish loading embedding!')

    # data loading
    train_df = pd.read_csv("./data/train_split.csv")
    train_df, val_df = train_test_split(train_df, test_size=VALID_SAMPLE_RATE)

    # resample 
    # 0.15 pos_neg_ratio, result f1 = 0.65, recall = 0.82
    train_df = resample(train_df, 0.15)

    th = 0.5
    step_per_epoch = 300
    epoches = 20

    model = quora_detector(embeddings_index['how'].shape[0], 5)
    train(model, train_df, val_df, learning_rate=0.001, batch_size=512, optimizer="Adam", iterations=epoches, threshold=th,
          step_per_epoch=step_per_epoch)

    model.eval()
    log = "thresold: "+str(th) + ", step per epoch: "+ str(step_per_epoch) + ", epoches: "+str(epoches) + "\n\n"

    # evaluation on training data
    print("Training prediction loading")
    train_part_df = pd.read_csv("./data/train_split.csv").sample(100000)
    train_log, train_f1 = evaluation(model, train_part_df)
    print(train_log)
    log += "Training:\n" + train_log + "\n"

    # evaluation on validation set
    print("Predition loading")
    val_log, val_f1 = evaluation(model, val_df)
    print(val_log)
    log += "Validation:\n" + val_log + "\n"

    # Final test
    print("Final Test")
    test_df = pd.read_csv("data/test_split.csv")
    test_log, test_f1 = evaluation(model, test_df)
    print(test_log)
    log += "Validation:\n" + test_log

    # save model and log into files
    with open("./model_" + str(np.round(test_f1, 4)), 'wb') as f:
        pickle.dump(model, f)
    with open("./log_" + str(np.round(test_f1, 4)) + ".txt", 'w') as f:
        f.write(log)