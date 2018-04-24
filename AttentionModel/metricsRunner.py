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
    
    #run metrics & get stats
    listOfStats = runMetrics(lab, pred, classToAnsMap,restoreModelPath)
    
    #save to pickle
    data = {}
    data['labels'] = lab
    data['preds'] = pred
    data['classToAnsMap'] = classToAnsMap
    dateID = restoreModelPath.split('/')[-2]
    saveToPickle(data, 'labpreds{}.pkl'.format(dateID))
    print('Metrics Completed.')
    
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
    
def getROCCurveMetrics():
    import matplotlib.pyplot as plt
    lab = [7, 2, 4,0,3,3]
    pred = [4,4,4,0,5,3]
    classes = np.arange(0,8)
    n_classes = len(classes)
    print(recall_score(lab, pred, labels=classes, average='micro'))
    print(recall_score(lab, pred, labels=classes, average='macro'))
    print(precision_score(lab, pred, labels=classes, average='micro'))
    print(precision_score(lab, pred, labels=classes, average='macro'))
    print(f1_score(lab, pred, labels=classes, average='micro'))
    print(f1_score(lab, pred, labels=classes, average='macro'))
    print(roc_curve(lab, pred, pos_label=0))
    
    y_truth = label_binarize(lab, classes=classes)
    y_score = label_binarize(pred, classes=classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_truth[:, i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_truth.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    lw=2
    #colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #for i, color in zip(range(n_classes), colors):
    #    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #             label='ROC curve of class {0} (area = {1:0.2f})'
    #             ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    
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
    
    metricsFileStats = 'MetricStatsQuAtt.csv'
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
    
    
    