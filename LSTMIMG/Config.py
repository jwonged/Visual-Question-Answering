'''
Created on 14 Jan 2018

@author: jwong
'''

class Config(object):
    '''
    Config file for LSTMIMG
    '''


    def __init__(self, params):
        pass
    
    nOutClasses = 1000
    batch_size = 20
        
    nTrainEpochs = 20
    dropoutVal = 0.2
    modelOptimizer = "GradDesc"
    lossRate = 0.1
    lossRateDecay = 0.9 #noDecay = 1
    max_gradient_norm = 5 #for clipping
    
    logFile = 'LSTMIMGresults.txt'
    
    trainEmbeddings = False
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    shortenedEmbeddingsFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddings.npz'
    fullDatasetVocab = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    
    idQnTrainFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/QnWordsIDlistTrain_OpenEnded.json'
    idValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/QnWordsIDlistValTest_OpenEnded.json'
    
    trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
    valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
    testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
    
    trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
    valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
    testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
    
    #other options currently manual
    #Sentence padding = pad to max len
    #Handling words not in embeddings = ignore (or give UNK vec)
    #LSTM input = 1st word as img or concat word+img on each time step

    
    
    