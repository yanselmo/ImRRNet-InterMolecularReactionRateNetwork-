import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from gensim.models import word2vec
from mol2vec import features



class mol2vec_preprocessing():
    """
    class for mol2vec embedding
    initialized with mol2vecmodel
    """
    def __init__(self, mol2vecmodel, dim = 300):
        """
        *args
        mol2vecmodel(str): path where mol2vec pickle is saved
        dim(int): dimension of mol2vec model
        """
        self.mol2vecmodel = word2vec.Word2Vec.load(mol2vecmodel)
        self.mol2vec_embedding_matrix = np.r_[self.mol2vecmodel.wv.syn0, np.zeros((1,dim))]
        self.dim = dim

    def _process_uncommon_word(self, mol2vec_corpus,maxlen = 117):
        """
        You can prepare mol2vec corpus mol2vec package
        Corpus in examples were prepared from *.smi file with radius=1, without uncommon threashold
        According to mol2vecmodel vocabulary, several low frequent tokens may be replaced by "UNK"
        This function makes length of corpus to maxlen by padding token "!".
        This "!" token will embedded to zeros vector by appending one zeros row 
            to embedding matrix of mol2vec model
        *args
        mol2vec_corpus(str): mol2vec corpus
        maxlen(int): maximum length of corpus
        """
        newcorpus = []
        for line in mol2vec_corpus:
            line = line.split('\n')[0]
            line = line.split(' ')
            sentence = []
            i = 0
            for word in line:
                if word not in self.mol2vecmodel.wv.vocab:
                    word = 'UNK'
                sentence.append(word)
                i+=1
            j = maxlen - i
            for _ in range(j):
                sentence.append('!')
            newcorpus.append(sentence)
        return newcorpus


    def _corpus2index(self, corpus):
        '''
        convet token to its index in vocabulary set of mol2vec model
        if token is not found in mol2vec model, index of UNK token is used.
        '''
        embedding_matrix = np.zeros((len(self.mol2vecmodel.wv.vocab), self.dim))
        newsentences = []
        for sentence in corpus:
            newsentence = []
            i = 0
            for token in sentence:
                if token in self.mol2vecmodel.wv.vocab.keys():
                    newsentence.append(self.mol2vecmodel.wv.vocab[token].index)
                else:
                    newsentence.append(len(embedding_matrix))
            newsentences.append(newsentence)
        newsentences = np.array(newsentences)
        return newsentences

    def convert_corpus_to_index(self, corpus, maxlen = 117):
        """
        preprocessing module
        get mol2vec corpus, and convert it to list of indices
        *args
        corpus(str): mol2vec corpus
        maxlen(int): maximum length of corpus
        """

        UNK_preprocess = self._process_uncommon_word(corpus, maxlen)
        indices = self._corpus2index(UNK_preprocess)
        return indices


if __name__ == "__main__":
    pass
