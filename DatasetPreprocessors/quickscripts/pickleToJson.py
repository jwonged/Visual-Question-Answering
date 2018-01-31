'''
Created on 18 Dec 2017

@author: jwong
'''
import pickle
import json

def convert():
    xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.pkl'
    yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.pkl'
    xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
    yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
    xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.pkl'
    yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.pkl'
    
    xTrainjson = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.json'
    yTrainjson = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.json'
    xValjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.json'
    yValjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.json'
    xTestjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.json'
    yTestjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.json'
    '''
    print('Reading ' + xTrainPickle)
    with open(xTrainPickle, 'rb') as pklFile:
        trainX = pickle.load(pklFile)
    print('Writing ' + xTrainjson)
    with open(xTrainjson, 'w') as outFile:
        json.dump(trainX, outFile)
    
    print('Reading ' + yTrainPickle)
    with open(yTrainPickle, 'rb') as pklFile:
        trainY = pickle.load(pklFile)
    print('Writing ' + yTrainjson)
    with open(yTrainjson, 'w') as outFile:
        json.dump(trainY, outFile)
        
    print('Reading ' + xValPickle)
    with open(xValPickle, 'rb') as pklFile:
        valX = pickle.load(pklFile)
    print('Writing ' + xValjson)
    with open(xValjson, 'w') as outFile:
        json.dump(valX, outFile)
        
    print('Reading ' + yValPickle)
    with open(yValPickle, 'rb') as pklFile:
        valY = pickle.load(pklFile)
    print('Writing ' + yValjson)
    with open(yValjson, 'w') as outFile:
        json.dump(valY, outFile)
    
    print('Reading ' + xTestPickle)
    with open(xTestPickle, 'rb') as pklFile:
        testX = pickle.load(pklFile)
    print('Writing ' + xTestjson)
    with open(xTestjson, 'w') as outFile:
        json.dump(testX, outFile)
    '''
    print('Reading ' + yTestPickle)
    with open(yTestPickle, 'rb') as pklFile:
        testY = pickle.load(pklFile)
    print('Writing ' + yTestjson)
    with open(yTestjson, 'w') as outFile:
        json.dump(testY, outFile)
        
    print('Completed.')

def splitTrainSet():
    xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.pkl'
    yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.pkl'
    
    xTrainPickle1 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx1.pkl'
    xTrainPickle2 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx2.pkl'
    xTrainPickle3 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx3.pkl'

    yTrainPickle1 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy1.pkl'
    yTrainPickle2 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy2.pkl'
    yTrainPickle3 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy3.pkl'

    
    data = readPickleFile(xTrainPickle)
    writePickleFile(data[:90000], xTrainPickle1)
    writePickleFile(data[90000:180000], xTrainPickle2)
    writePickleFile(data[180000:], xTrainPickle3)
    
    del data[:]
    data = readPickleFile(yTrainPickle)
    writePickleFile(data[:90000], yTrainPickle1)
    writePickleFile(data[90000:180000], yTrainPickle2)
    writePickleFile(data[180000:], yTrainPickle3)
    
def readPickleFile(fileName):
    print('Reading ' + fileName)
    with open(fileName, 'rb') as pklFile:
        return pickle.load(pklFile)    

def writePickleFile(data, fileName):
    print('Writing ' + fileName)
    with open(fileName, 'wb') as pklFile:
        return pickle.dump(data, pklFile) 

def createMiniSet():
    xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
    yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
    miniX = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainX.pkl'
    miniY = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainY.pkl'
    miniValX = '/media/jwong/Transcend/VQADataset/DummySets/miniValX.pkl'
    miniValY = '/media/jwong/Transcend/VQADataset/DummySets/miniValY.pkl'
    
    print('Reading ' + xValPickle)
    with open(xValPickle) as jFile:
        data = pickle.load(jFile)
    print('Writing ' + miniX)
    with open(miniX, 'w') as outFile:
        pickle.dump(data[:1000], outFile)
    print('Writing ' + miniValX)
    with open(miniValX, 'w') as outFile:
        pickle.dump(data[1000:1200], outFile)
        
    print('Reading ' + yValPickle)
    with open(yValPickle) as jFile:
        data = pickle.load(jFile)
    print('Writing ' + miniY)
    with open(miniY, 'w') as outFile:
        pickle.dump(data[:1000], outFile)
    print('Writing ' + miniValY)
    with open(miniValY, 'w') as outFile:
        pickle.dump(data[1000:1200], outFile)
    
    print('Completed.')



if __name__ == "__main__":
    splitTrainSet()