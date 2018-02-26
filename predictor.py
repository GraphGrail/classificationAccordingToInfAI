
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.contrib import learn
from keras.models import load_model

import numpy as np

from textProcessing.textProcessor import TextProcessor

def convertToOneHot(wordIdsArrays, vocabularySize):
    numberOfDocs = wordIdsArrays.shape[0]
    rowLength = wordIdsArrays.shape[1]
    res = np.zeros(shape=(numberOfDocs, vocabularySize - 1), dtype=int)
    i = 0
    while i < numberOfDocs:
        g = 0
        while g < rowLength:
            if wordIdsArrays[i][g] != 0:
                res[i][wordIdsArrays[i][g] - 1] += 1
            g += 1
        i += 1
    return res

class Predictor:
    def __init__(self):
        temp = None
        with open("thematicsId", "r") as f:
            temp = eval(f.read())
            
        self.__idThematics = dict((temp[k], k) for k in temp)
        
        self.__textProcessor = TextProcessor()
        
        self.__vocab = learn.preprocessing.VocabularyProcessor.restore("./vocab")
        
        self.__model = load_model("model")
    def predict(self, documents, resultAsClassNumber=True):
        
        X = self.__textProcessor.convertSequenceOfDocuments(documents)
        X = [" ".join(i) for i in X]

        X = np.array(list(self.__vocab.transform(X)))
        
        X = convertToOneHot(X, len(self.__vocab.vocabulary_._mapping))
        
        prediction = self.__model.predict(X)
        
        predictedClassNumbers = prediction.argmax(axis=1)
        
        if resultAsClassNumber == True:
            return predictedClassNumbers
        else:
            res = np.array([self.__idThematics[x] for x in predictedClassNumbers])
            return res
        
    def getIdthematics(self):
        return self.__idThematics
            
    __idThematics = None
    __textProcessor = None
    __vocab = None
    __model = None


