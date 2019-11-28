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

from utils import preprocess
from utils import getData

torch.manual_seed(123)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class quora_detector(nn.Module):
    def __init__(self, vec_dim):
        super(quora_detector, self).__init__()
        self.rnn_1 = nn.LSTM(input_size = vec_dim, hidden_size = 64, batch_first = True, bidirectional = True)
        self.rnn_2 = nn.LSTM(input_size = 128, hidden_size = 64, batch_first = True, bidirectional = True)
        
        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1)
        #self.stat_dense = nn.Linear(fea_num, 64)
        self.cated_dense = nn.Linear(128, 128)
        self.dropout = nn.Dropout(p = 0.5)
        self.batchnorm = nn.BatchNorm1d(128)
        
        self.final_dense = nn.Linear(64, 1)
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

    def forward(self, seq, length):
        # context embedding
        '''
        # spatial dropout(0.5)
        seq = seq.permute(0,2,1)
        seq = F.dropout2d(seq, p = 0.5, training=True, inplace=False)
        seq = seq.permute(0,2,1)
        '''
        #seq = torch.nn.utils.rnn.pack_padded_sequence(seq, length, batch_first = True)

        context, _ = self.rnn_1(seq)
        #context = self.batchnorm_1(context)
        context, _ = self.rnn_2(context)
        #context, _ = torch.nn.utils.rnn.pad_packed_sequence(context, batch_first = True)
        #context = self.last_element(context, length)
        context = self.batchnorm(context[:,-1,:])[:,:,None]
        
        context = context.permute(0,2,1)
        # 1d convolution layer, 64 convolution kernels(to batchsize*64*128)
        context = self.conv(context)
        # global max pooling(to batchsize*64(*1))
        context = F.max_pool1d(context, kernel_size = context.shape[2:])
        context = context[:,:,-1]
        context = F.relu(context)
        output = context

        '''
        # stats features embedding(to batchsize*64)
        stat_features = self.stat_dense(stat_features)
        stat_features = F.relu(stat_features)
        #stat_features = self.dropout(stat_features)

        # concat two embeddings(to batchsize*128)
        output = torch.cat((context, stat_features), 1)
        '''
        '''
        # full connection layer with dropout(0.5) and batchnorm layer(to batchsize*128)
        output = self.cated_dense(context)
        #output = self.batchnorm(output)
        output = F.relu(output)
        #output = self.dropout(output)
        '''
        # final output, 2 classes(to batchsize*2)
        output = self.final_dense(context)
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
    text = text[:-1].split()[:30]
    embeds = []
    for x in text:
        if x in embeddings_index:
            embeds.append(embeddings_index[x])
        else:
            embeds.append(np.zeros(300))
    # embeds = [embeddings_index.get(x, empyt_emb) for x in text]
    embeds+= [empyt_emb] * (30 - len(embeds))
    return np.array(embeds)

def predict(model, sentences, batch_size = 100, threshold = 0.5, pred_bar = True):
    sentences = np.array(sentences)
    length = torch.tensor(list(map(len, sentences)))
    ret = np.array([])
    iterator = tqdm(range((len(sentences)-1)//batch_size + 1)) if pred_bar == True else range((len(sentences)-1)//batch_size + 1)
    for k in iterator:
        batch_L = length[k*batch_size:(k+1)*batch_size]
        batch_L, idx = torch.sort(batch_L, descending = True)
        batch_L = batch_L.to(DEVICE)

        batch_sentences = sentences[k*batch_size:(k+1)*batch_size][idx]
        
        batch_X = torch.tensor([text_to_array(sentence) for sentence in batch_sentences]).float().to(DEVICE)
        #batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)

        logit = model.forward(batch_X, batch_L)[:,0]

        prob = prob = torch.sigmoid(logit)
        
        # recover the order the data
        _, rev_idx = torch.sort(idx)
        res = torch.tensor(np.where(prob.cpu() > threshold, 1, 0))[rev_idx]
        ret = np.r_[ret, res]
    return ret

def train(model, train_df, val_df, learning_rate = 0.001, batch_size = 100, optimizer = "Adam", iterations = 100, threshold = 0.5, setp_per_epoch = 1000):
    train_sentences = np.array(train_df['question_text'])
    train_targets = torch.tensor(np.array(train_df['target']))
    train_length = torch.tensor(list(map(len, train_sentences)))

    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # put model on GPU and parallelize
    model.to(DEVICE)
    #model = nn.DataParallel(model)

    print("Training will on GPU" if DEVICE == torch.device("cuda:0") else "Training will on CPU")

    losses = []
    #train_errors = []
    #test_errors = []

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    train_start_time = time.perf_counter()
    val_show = val_df[:3000]
    for epoch in range(iterations):
        idx = np.random.permutation(train_sentences.shape[0])
        train_sentences = train_sentences[idx]
        train_targets = train_targets[idx]
        step_per_epoch = min(setp_per_epoch, (len(train_sentences)-1)//batch_size + 1)
        with tqdm(total = setp_per_epoch, desc='epoch {}'.format(epoch+1)) as t:
            for k in range(setp_per_epoch):
                # free unreferenced cuda memory and put gradient zero
                torch.cuda.empty_cache()
                optimizer.zero_grad()

                batch_L = train_length[k*batch_size:(k+1)*batch_size]
                batch_L, idx = torch.sort(batch_L, descending = True)
                batch_L = batch_L.to(DEVICE)

                batch_sentences = train_sentences[k*batch_size:(k+1)*batch_size][idx]
                
                batch_X = torch.tensor([text_to_array(sentence) for sentence in batch_sentences]).float().to(DEVICE)
                #batch_X = torch.nn.utils.rnn.pad_sequence(batch_X, batch_first=True).float().to(DEVICE)
                batch_Y = train_targets[k*batch_size:(k+1)*batch_size][idx].float().to(DEVICE)

                logit = model.forward(batch_X, batch_L)[:,0]
                loss = criterion(logit, batch_Y)

                loss.backward()
                optimizer.step()

                losses.append(float(loss))
                t.set_postfix(loss='{:^7.3f}'.format(loss))
                t.update()
        #print("last loss in epoch {}: {}".format(epoch+1, loss))
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

if __name__ == "__main__":
    VALID_SAMPLE_RATE = 0.1

    # embedding data loading
    embeddings_index = KeyedVectors.load("./embedding/glove.840B.300d.gensim")
    print('finish loading embedding!')

    train_df = pd.read_csv("./data/train_split.csv")
    train_df, val_df = train_test_split(train_df, test_size=VALID_SAMPLE_RATE)
    #train_df = train_df.iloc[:100000]
    #val_df = val_df.iloc[:1000]

    for th in [0.5]:
        
        model = quora_detector(embeddings_index['how'].shape[0])
        train(model, train_df, val_df, \
            learning_rate = 0.0001, batch_size = 128, optimizer = "Adam", iterations = 20, threshold = th, setp_per_epoch = 1000)
        
        '''
        with open("./model", 'rb') as f:
            model = pickle.load(f)
        '''
        model.eval()
        # Final test
        test_df = pd.read_csv("data/test_split.csv")
        prediction = predict(model, test_df['question_text'], batch_size = 512)
        print("Evaluation loading")
        acc = accuracy_score(test_df['target'], prediction)
        f1 = f1_score(test_df['target'], prediction)
        print("Accuracy: {}".format(acc))
        print("f1 score: {}".format(f1))
        fpr, tpr, _ = metrics.roc_curve(test_df['target'], prediction)
        print("AUC: {}".format(metrics.auc(fpr, tpr)))
        print("Accuracy on ground truth 0(TNR): {}".format(1-fpr[1]))
        print("Accuracy on ground truth 1(TPR,recall): {}".format(tpr[1]))
        print("Balanced accuracy: {}".format((1-fpr[1] + tpr[1])/2))