
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

class PredictionResult:
    def __init__(self):
        self.status = -1
        self.errorMessage = "Не достаточно данных"
    status = None # 0 - all ok, -1 - error
    topic = None
    weight = None
    errorMessage = None
    
class Topic:
    topicId = None
    topicName = None
    
    
class Predictor:
    def __init__(self):
        temp = None
        with open("thematicsId", "r") as f:
            temp = eval(f.read())
            
        self.__idThematics = dict((temp[k], k) for k in temp)
        
        self.__textProcessor = TextProcessor()
        
        self.__vocab = learn.preprocessing.VocabularyProcessor.restore("./vocab")
        
        self.__model = load_model("model")
    def predict(self, documents):
        
        if type(documents) != list:
            raise ValueError("Input value 'documents' is not a list")
            
        docNum = len(documents)
        appropriateString = [False] * docNum
        i = 0
        while i < docNum:
            if len(documents[i]) > 2:
                appropriateString[i] = True
            i += 1
            
        X = self.__textProcessor.convertSequenceOfDocuments(documents)
        X = [" ".join(i) for i in X]

        X = np.array(list(self.__vocab.transform(X)))
        
        X = convertToOneHot(X, len(self.__vocab.vocabulary_._mapping))
        
        prediction = self.__model.predict(X)
        
        predictedClassNumbers = prediction.argmax(axis=1)
        weights = [prediction[x][predictedClassNumbers[x]] for x in range(docNum)]
        predictedThematicTexts = [self.__idThematics[x] for x in predictedClassNumbers]
        
        res = [PredictionResult() for x in range(docNum)]
        
        i = 0
        while i < docNum:
            if appropriateString[i] == True:
                res[i].status = 0
                topic = Topic()
                topic.topicId = predictedClassNumbers[i]
                topic.topicName = predictedThematicTexts[i]
                res[i].topic = topic
                res[i].weight = weights[i]
                res[i].errorMessage = ""
            i += 1
            
        return res
        
    def getIdthematics(self):
        return self.__idThematics
            
    __idThematics = None
    __textProcessor = None
    __vocab = None
    __model = None


