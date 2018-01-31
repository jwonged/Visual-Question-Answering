'''
Created on 19 Dec 2017

@author: jwong
'''
import pickle
import csv
import pandas as pd
import json
    
def createMiniSet():
    miniX = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainX.pkl'
    miniY = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainY.pkl'
    miniValX = '/media/jwong/Transcend/VQADataset/DummySets/miniValX.pkl'
    miniValY = '/media/jwong/Transcend/VQADataset/DummySets/miniValY.pkl'
    
    xtraincsv = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainX.csv'
    ytraincsv = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainY.csv'
    
    xtrain = readPickleFile(miniX)
    ytrain = readPickleFile(miniY)
    writeCSVFile(xtrain, xtraincsv)
    writeCSVFile(ytrain, ytraincsv)
    
    print('Completed.')

def readPickleFile(fileName):
    print('Reading ' + fileName)
    with open(fileName, 'rb') as pklFile:
        return pickle.load(pklFile)

def writeCSVFile(data, fileName):
    print('Writing ' + fileName)
    with open(fileName, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for item in data:
            writer.writerow(item)
            
index = 0
def getbatch():
    global index
    xtraincsv = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainX.csv'
    ytraincsv = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainY.csv'
    
    reader = pd.read_table(xtraincsv, sep=',', chunksize=4)
    return reader

def readTest():
    data = getbatch()
    
    for items in data:
        for item in items:
            print(item)
            print('length of items in the batch: '.format(len(item)))
        break
    
def checkVecAvailable():
    import gensim
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(pretrainedw2v, binary=True)
    word = '?'
    print(model.wv[word])

def checkSentence():
    from nltk import word_tokenize
    sentence = 'What is the boy kicking?'
    print(word_tokenize(sentence))
    
def checkFileContent():
    qnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    valTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
    
    with open(qnTrain) as annotFile:
        annotBatch = json.load(annotFile)
    
    for i, (key, val) in enumerate(annotBatch.iteritems()):
        if i > 5:
            break
        print(key)
        print(val)
    
if __name__ == '__main__':    
    checkFileContent()