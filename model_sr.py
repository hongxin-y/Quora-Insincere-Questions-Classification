import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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
        self.maxPool = torch.max

        self.fc1 = nn.Linear(128+fea_num, 64)
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
        context = nn.utils.rnn.pack_padded_sequence(seq, length, batch_first=True)
        context, _ = self.rnn_1(context)
        context, _ = self.rnn_2(context)
        context, _ = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)
        context, _ = self.maxPool(context, 1)
        context = torch.cat((context, statistics), 1)
        context = self.fc1(context)
        context = self.relu1(context)
        context = self.bn1(context)
        context = self.fc2(context)
        context = self.relu2(context)
        context = self.bn2(context)

        output = self.out(context)
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
    # list(map(len, train_sentences))
    # train_stats_features =

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
    val_show = val_df[:3000]
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


if __name__ == "__main__":
    VALID_SAMPLE_RATE = 0.1

    embeddings_index = KeyedVectors.load("./embeddings/glove.840B.300d.gensim")
    print('finish loading embedding!')

    train_df = pd.read_csv("./data/train_split.csv")
    train_df, val_df = train_test_split(train_df, test_size=VALID_SAMPLE_RATE)

    th = 0.5
    step_per_epoch = 300
    epoches = 20

    model = quora_detector(embeddings_index['how'].shape[0], 5)
    train(model, train_df, val_df, learning_rate=0.001, batch_size=512, optimizer="Adam", iterations=epoches, threshold=th,
          step_per_epoch=step_per_epoch)

    model.eval()
    print("Evaluation start with threshold {}, step per epoch {}, iterations number {}".format(th,
                                                                                               step_per_epoch,
                                                                                               epoches))

    print("Training prediction loading")
    train_df = pd.read_csv("./data/train_split.csv").sample(100000)
    prediction = predict(model, train_df['question_text'], batch_size=512)
    acc = accuracy_score(train_df['target'], prediction)
    f1 = f1_score(train_df['target'], prediction)
    print("Accuracy: {}".format(acc))
    print("f1 score: {}".format(f1))
    fpr, tpr, _ = metrics.roc_curve(train_df['target'], prediction)
    print("AUC: {}".format(metrics.auc(fpr, tpr)))
    print("Accuracy on ground truth 0(TNR): {}".format(1 - fpr[1]))
    print("Accuracy on ground truth 1(TPR,recall): {}".format(tpr[1]))
    print("Balanced accuracy: {}".format((1 - fpr[1] + tpr[1]) / 2))

    # evaluation on validation set
    print("Predition loading")
    prediction = predict(model, val_df['question_text'], batch_size=512)
    acc = accuracy_score(val_df['target'], prediction)
    f1 = f1_score(val_df['target'], prediction)
    print("Accuracy: {}".format(acc))
    print("f1 score: {}".format(f1))
    fpr, tpr, _ = metrics.roc_curve(val_df['target'], prediction)
    print("AUC: {}".format(metrics.auc(fpr, tpr)))
    print("Accuracy on ground truth 0(TNR): {}".format(1 - fpr[1]))
    print("Accuracy on ground truth 1(TPR,recall): {}".format(tpr[1]))
    print("Balanced accuracy: {}".format((1 - fpr[1] + tpr[1]) / 2))

    # Final test
    print("Evaluation loading")
    test_df = pd.read_csv("data/test_split.csv")
    prediction = predict(model, test_df['question_text'], batch_size=512)
    acc = accuracy_score(test_df['target'], prediction)
    f1 = f1_score(test_df['target'], prediction)
    print("Accuracy: {}".format(acc))
    print("f1 score: {}".format(f1))
    fpr, tpr, _ = metrics.roc_curve(test_df['target'], prediction)
    print("AUC: {}".format(metrics.auc(fpr, tpr)))
    print("Accuracy on ground truth 0(TNR): {}".format(1 - fpr[1]))
    print("Accuracy on ground truth 1(TPR,recall): {}".format(tpr[1]))
    print("Balanced accuracy: {}".format((1 - fpr[1] + tpr[1]) / 2))

    # save model into binary file
    with open("./model_" + str(np.round(f1, 4)), 'wb') as f:
        pickle.dump(model, f)