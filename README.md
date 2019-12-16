# Quora Insincere Questions Classification

In this project, we try to implement some neural networks to identify toxic and divisive questions on Quora. We construct two different architectures and try to use bootstrapping resampling, statistical features, different padding methods and spatial dropout to improve performance. The first model is slightly more complicated, but it works very well; another one simplifies the previous model, but still maintains a relatively good result. We also have found a baseline from Kaggle whose F1-score is 0.62. After tuning our models, the F1-score of them is about 0.65. 

[kaggle address](https://www.kaggle.com/c/quora-insincere-questions-classification/overview)

## Embedding Turotial

[Download Embedding](https://www.kaggle.com/iezepov/gensim-embeddings-dataset)

Download pre-trained embeddings and make sure "glove.840B.300d.gensim", "glove.840B.300d.gensim.vectors.npy", "paragram_300_sl999.gensim" and "paragram_300_sl999.gensim.vectors.npy" are under the ./embeddings

Put "train.csv" and "test.csv" from kaggle under ./data

## Train Model

Split training set into the new training set and test set using:

```python
python splitData.py
```

Train the model.

```
python model.py
```

or

```
python model_sr.py
```

