'''
Created on 24 Apr 2018

@author: jwong
'''

from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from TrainProcessor import LSTMIMGProcessor, TestProcessor
import numpy as np
from scipy.stats import ttest_ind
from scipy.special import stdtr
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from collections import Counter
import argparse
import pickle
import csv

def runMetricsForInternalTestSet(args, restoreModel, restoreModelPath):
    #config = LSTMIMG_LapConfig(load=True, args)
    config = LSTMIMG_GPUConfig(load=True, args=args)
    
    print('Running Validation Test on Model')
    valTestReader = LSTMIMGProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    model = LSTMIMGmodel(config)
    model.loadTrainedModel(restoreModel, restoreModelPath)
    lab, pred, classToAnsMap = model.runEvaluationMetrics(valTestReader)
    model.destruct()
    valTestReader.destruct()
    
    #run metrics & get stats
    listOfStats = runMetrics(lab, pred, classToAnsMap,restoreModelPath)
    
    #save to pickle
    data = {}
    data['labels'] = lab
    data['preds'] = pred
    data['classToAnsMap'] = classToAnsMap
    dateID = restoreModelPath.split('/')[-2]
    saveToPickle(data, 'labpreds{}.pkl'.format(dateID))
    print('predictions made')
    
    return listOfStats

def runMetrics(lab, pred, classToAnsMap, pathToModel):
    #createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 1000)
    createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 20)
    createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 15)
    
    classes = classToAnsMap.keys()
    
    listOfStats = []
    listOfStats.append(pathToModel)
    
    msg = 'micro recall: {}'.format(
        recall_score(lab, pred, labels=classes, average='micro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'macro recall: {}'.format(
        recall_score(lab, pred, labels=classes, average='macro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'micro precision: {}'.format(
        precision_score(lab, pred, labels=classes, average='micro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'macro precision: {}'.format(
        precision_score(lab, pred, labels=classes, average='macro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'micro f1 score: {}'.format(
        f1_score(lab, pred, labels=classes, average='micro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'macro f1 score: {}'.format(
        f1_score(lab, pred, labels=classes, average='macro'))
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    msg = 'mcc: {}'.format(
        matthews_corrcoef(lab, pred) )
    #logStats.writerow([msg])
    listOfStats.append(msg)
    
    print('stats logging complete.')
    return listOfStats
    

def createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, num):
    dateID = pathToModel.split('/')[-2]
    metricsFileCF = 'MetricConfMatrix{}_{}.csv'.format(num,dateID)
    print('Preparing confusion matrix for {}'.format(metricsFileCF))
    fconf = open(metricsFileCF, 'wb')
    logConf = csv.writer(fconf)
    
    ansTups = Counter(lab).most_common(num+1)
    classes = []
    for tup in ansTups:
        if tup[0] != -1 and tup[0] != '-1':
            classes.append(tup[0])
    
    print('classes: {}'.format(len(classes)))
    
    classesAsNums = classes
    classesAsStrings = [classToAnsMap[i] for i in classesAsNums]
    
    #logging
    logConf.writerow([pathToModel])
    logConf.writerow(classesAsStrings)
    logConf.writerow(classesAsNums)
    
    #create matrix
    confmatr = confusion_matrix(lab, pred, labels=classesAsNums)
    for row in confmatr:
        logConf.writerow(row)
    fconf.close()

def saveToPickle(data, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to {}'.format(fileName))
    
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--lstmtype', choices=['bi', 'uni'], default='bi')
    parser.add_argument('--useuntrainedembed', help='use untrained embeddings', action='store_false')
    parser.add_argument('--donttrainembed', help='use untrained embeddings', action='store_false')
    parser.add_argument('--useconcat', help='use concat instead of elmult', action='store_false')
    parser.add_argument('--noshuffle', help='Do not shuffle dataset', action='store_false')
    parser.add_argument('--mmAtt', help='Use multimodal attention', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    
    metricsFileStats = 'MetricStatsLSTMIMG.csv'
    print('Results will be logged to {}'.format(metricsFileStats))
    fstats = open(metricsFileStats, 'wb')
    logStats = csv.writer(fstats)
    
    #QuAtt
    modelFile = 'results/LSTM1Apr1-34/LSTM1Apr1-34.meta'
    modelPath = 'results/LSTM1Apr1-34/'
    stats1 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    modelFile = 'results/LSTM1Apr4-9/LSTM1Apr4-9.meta'
    modelPath = 'results/LSTM1Apr4-9/'
    stats2 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    modelFile = 'results/LSTM1Apr6-37/LSTM1Apr6-37.meta'
    modelPath = 'results/LSTM1Apr6-37/'
    stats3 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    
    for msg in stats1:
        logStats.writerow([msg])
    for msg in stats2:
        logStats.writerow([msg])
    for msg in stats3:
        logStats.writerow([msg])
    
    fstats.close()
    print('Main Complete.')
    