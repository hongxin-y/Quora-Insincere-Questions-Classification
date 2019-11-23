import numpy as np
import pandas as pd
import nltk
import pickle
from gensim.models import KeyedVectors


def get_index2word_sentence2index():
    index2word = {}  # 从索引对应到word {1: "the", 2: "table"}
    word2index = {}  # 从word到索引 {"the": 1, "table": 2}
    sentence2index = []  # 从句子对应到向量 "the table" -> [1, 2]
    targets = [] # 类别

    # 字符串长度 词的数量 大写字母数量 标点符号数量 特殊词数量 句子中token的数量(向量化词长度)
    statistics_feature = []

    index_count = 0
    special_words = ["stupid", "discriminat", "dictatorship", "black", "gay", "race", "penis", "hypocritical", "kill",
                     "muslim", "sex", "political", "fuck", "dick", "vagina"]

    data = pd.read_csv("./data/train.csv")

    for i in range(data.shape[0]):
        sentence = data.iloc[i, 1]
        target = data.iloc[i, 2]

        tmp_stat_feature = []
        tmp_sentence2index = []

        # 字符串长度
        tmp_stat_feature.append(len(sentence))
        # 词数量
        tmp_stat_feature.append(len(sentence.split(" ")))
        # 判断大写字母数量和标点数量
        cap_count = 0
        punc_count = 0
        for c in sentence:
            if c >= "A" and c <= "Z":
                cap_count += 1
            if c != " " and (c < "0" or (c > "9" and c < "A") or (c > "Z" and c < "a") or c > "z"):
                punc_count += 1

        tmp_stat_feature.append(cap_count)
        tmp_stat_feature.append(punc_count)
        # 判断特殊词
        sentence = sentence.lower()
        special_words_count = 0
        for special_word in special_words:
            special_words_count += sentence.count(special_word)

        tmp_stat_feature.append(special_words_count)

        # 将文本转化为索引
        sentence = sentence.replace("'re", " are")
        sentence = sentence.replace("n't", " not")
        sentence = sentence.replace("what's", "what is")
        sentence = sentence.replace("that's", "that is")
        tokens = nltk.word_tokenize(sentence)
        tmp_stat_feature.append(len(tokens))

        for token in tokens:
            if token not in word2index.keys():
                index2word[index_count] = token
                word2index[token] = index_count
                index_count += 1

            tmp_sentence2index.append(word2index[token])

        sentence2index.append(tmp_sentence2index)
        statistics_feature.append(tmp_stat_feature)
        targets.append(target)

    return word2index, index2word, sentence2index, statistics_feature, targets


# 生成词向量
# 根据word2index中已经存在的word做词向量
# 如果glove和paragram中都有,则glove*0.7 + paragram*0.3
# 如果仅glove中有,则只保存glove
# 其余情况为np.zeros(300)
def get_word2vec(word2index):
    word2vec = {}
    model_glove = KeyedVectors.load("./embedding/glove.840B.300d.gensim")
    model_paragram = KeyedVectors.load("./embedding/paragram_300_sl999.gensim")

    for key in word2index.keys():
        if key in model_glove and key in model_paragram:
            word2vec[key] = model_glove[key] * 0.7 + model_paragram[key] * 0.3

    return word2vec


# 得到每个句子最终的词向量
# sentence2index 为 [[1, 2]]时,生成[[vec1, vec2]] 词向量维度为300,句子间无padding
def get_index2vec(sentence2index, index2word, word2vec):
    sentenceVec = []
    for indexList in sentence2index:
        tmpVec = []
        for index in indexList:
            if index not in word2vec.keys():
                tmpVec.append(np.zeros(300))
            else:
                word = index2word[index]
                vec = word2vec[word]
                tmpVec.append(vec)
        sentenceVec.append(tmpVec)
    return sentenceVec


def preprocess():
    # 处理:
    word2index, index2word, sentence2index, statistics_feature, targets = get_index2word_sentence2index()
    with open("./data/word2index", "wb") as f:
        pickle.dump(word2index, f)
    with open("./data/index2word", "wb") as f:
        pickle.dump(index2word, f)
    with open("./data/sentence2index", "wb") as f:
        pickle.dump(sentence2index, f)
    with open("./data/statistics_feature", "wb") as f:
        pickle.dump(statistics_feature, f)
    with open("./data/targets", "wb") as f:
        pickle.dump(targets, f)

    del index2word, sentence2index, statistics_feature, targets

    word2vec = get_word2vec(word2index)
    with open("./data/word2vec", "wb") as f:
        pickle.dump(word2vec, f)


def getData():
    # 使用: 获取句子的词向量 [[vec11, vec12, ...], [vec21, vec22, ...], ...]
    # 未经padding
    with open("./data/sentence2index", "rb") as f:
        sentence2index = pickle.load(f)
    with open("./data/index2word", "rb") as f:
        index2word = pickle.load(f)
    with open("./data/word2vec", "rb") as f:
        word2vec = pickle.load(f)

    sentenceVec = get_index2vec(sentence2index, index2word, word2vec)

    del sentence2index, index2word, word2vec

    # 获取句子的target  [0, 1, ...]
    with open("./data/targets", "rb") as f:
        targets = pickle.load(f)

    # 获取statistic info  [[81, 32, 3, 2, 1, 45], [81, 32, 3, 2, 1, 42], ...]
    # 字符串长度 词的数量 大写字母数量 标点符号数量 特殊词数量 句子中token的数量(向量化词长度)
    with open("./data/statistics_feature", "rb") as f:
        statistics_feature = pickle.load(f)

    return sentenceVec, targets, statistics_feature


if __name__ == "__main__":
    preprocess()
    # 如果之前做过一次preprocess,之后都不需要调用preprocess,直接getData
    sentenceVec, targets, statistics_feature = getData()
    print("Finished!")
