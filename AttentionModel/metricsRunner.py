'''
Created on 24 Apr 2018

@author: jwong
'''

from model.Image_AttModel import ImageAttentionModel
from model.Qn_AttModel import QnAttentionModel
from configs.Attention_LapConfig import Attention_LapConfig
from configs.Attention_GPUConfig import Attention_GPUConfig
from utils.TrainProcessors import TestProcessor, AttModelInputProcessor
import numpy as np
from scipy.stats import ttest_ind
from scipy.special import stdtr
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
from collections import Counter
import argparse
import pickle
import csv

def runMetricsForInternalTestSet(args, restoreModel, restoreModelPath):
    print('Running metrics for model: {}'.format(restoreModel))
    
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    
    print('Running Validation Test on Model')
    valTestReader = AttModelInputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    #restoreModel = config.restoreModel
    #restoreModelPath = config.restoreModelPath
    
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)
    model.loadTrainedModel(restoreModel, restoreModelPath)
    
    lab, pred, classToAnsMap = model.runEvaluationMetrics(valTestReader)
    
    model.destruct()
    valTestReader.destruct()
    print(len(classToAnsMap))
    print(classToAnsMap[0])
    
    listOfStats = runMetrics(lab, pred, classToAnsMap,restoreModelPath)
    data = {}
    data['labels'] = lab
    data['preds'] = pred
    data['classToAnsMap'] = classToAnsMap
    dateID = restoreModelPath.split('/')[-1]
    saveToPickle(data, 'labpreds{}.pkl'.format(dateID))
    
    print('Metrics Completed.')
    return listOfStats
    
def runMetrics(lab, pred, classToAnsMap, pathToModel):
    createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 1000)
    createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 20)
    createConfusionMatrix(lab, pred, classToAnsMap, pathToModel, 15)
    
    classes = np.arange(0,len(classToAnsMap)-1)
    classesAsStrings = [classToAnsMap[i] for i in classes]
    logStats.writerow([pathToModel])
    
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
    dateID = pathToModel.split('/')[-1]
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
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('--attfunc', choices=['sigmoid', 'softmax'], default='softmax')
    parser.add_argument('-a', '--action', choices=['otest', 'val'], default='val')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('--notopk', help='No loading topk', action='store_true')
    parser.add_argument('--noqnatt', help='No loading qnAtt', action='store_true')
    parser.add_argument('--debugmode', help='Trace printing', action='store_true')
    parser.add_argument('--stackAtt', help='Trace printing', action='store_true')
    parser.add_argument('--attComb', choices=['concat', 'mult', 'add'], default='concat')
    parser.add_argument('--qnAttf', choices=['softmax', 'sigmoid'], default='sigmoid')
    parser.add_argument('--mmAtt', help='Use multimodal attention', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    
    metricsFileStats = 'MetricStats.csv'
    print('Results will be logged to {}'.format(metricsFileStats))
    fstats = open(metricsFileStats, 'wb')
    logStats = csv.writer(fstats)
    
    #QuAtt
    modelFile = 'results/Att29Mar22-27/att29Mar22-27.meta'
    modelPath = 'results/Att29Mar22-27/'
    stats1 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    modelFile = 'results/Att22Mar0-12/att22Mar0-12.meta'
    modelPath = 'results/Att22Mar0-12/'
    stats2 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    modelFile = 'results/Att27Mar19-42/att27Mar19-42.meta'
    modelPath = 'results/Att27Mar19-42/'
    stats3 = runMetricsForInternalTestSet(args, modelFile, modelPath)
    
    for msg in stats1:
        logStats.writerow([msg])
    for msg in stats2:
        logStats.writerow([msg])
    for msg in stats3:
        logStats.writerow([msg])
    
    fstats.close()
    print('Main Complete.')
    #ImAtt
    
    
    