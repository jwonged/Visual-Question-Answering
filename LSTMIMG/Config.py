'''
Created on 14 Jan 2018

@author: jwong
'''

class Config(object):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self):
        pass
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'imageAfterLSTM'
    
    nOutClasses = 1000  #1000 or 17140
    batch_size = 32
    imgVecSize = 1024
    wordVecSize = 300
    
    nTrainEpochs = 30
    nEpochsWithoutImprov = 3
    dropoutVal = 0.5
    modelOptimizer = "adam"
    lossRate = 0.001 #0.001 for adam, 0.01 for gradDesc
    lossRateDecay = 1 #noDecay = 1
    max_gradient_norm = -1 #for clipping
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    shuffle = True
    trainEmbeddings = True
    
    csvResults = '/media/jwong/Transcend/VQAresults/LSTMIMG/DummyPred_mainResult2601LSTMIMG_adamConcatAfter.csv'
    predictionsFile = '/media/jwong/Transcend/VQAresults/LSTMIMG/LSTMIMGpreds2501.txt'
    logFile = '/media/jwong/Transcend/VQAresults/LSTMIMG/DummyRes_mainResult2601LSTMIMG_adamConcatAfter.csv'
    saveModelFile = '/media/jwong/Transcend/VQADataset/DummySets/LSTMIMG-proto'
    
    '''Pickle file Contains:
            singleCountWords , wordToIDmap, 
            classToAnsMap, ansToClassMap (not yet)
    '''
    preprocessedVQAMapsFile = '/media/jwong/Transcend/VQADataset/preprocessedVQAmaps.pkl'
    
    
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    shortenedEmbeddingsWithUNKFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddingsWithUNK.npz'
    
    #empty for now
    datasetVocabWithUNK = '/media/jwong/Transcend/VQADataset/FullVQAVocabWwithUNK.txt'
    
    #Qn_ID --> 'qn'
    rawQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    rawQnValTestFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
    
    #Qn_ID --> id, id, id...
    idQnTrainFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/QnWordsIDlistTrain_OpenEnded.json'
    idValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/QnWordsIDlistValTest_OpenEnded.json'
    
    '''
    Annotations file in list with resolved single answer
    [
        {
            "question_id" : int,
            "image_id" : int,
            "question_type" : str,
            "answer_type" : str,
            "answers" : answer(str),
            "multiple_choice_answer" : str
        }, ...
    ]
    248,349 train
    60,753 val
    60,756 test
    '''
    trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
    valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
    testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
    
    #img_id --> img feature vec (1024)
    trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
    valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
    testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
    
    ansClass1000File = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
    mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/allTrainAnswers.csv'
    
    #contains vocab from train/val/test questions and answers
    vocabFile = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    fullDatasetVocab2301 = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    shortenedEmbeddingsWithoutUNKFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddings.npz'

    #other options currently manual
    #Sentence padding = pad to max len
    #Handling words not in embeddings = ignore (or give UNK vec)
    #LSTM input = 1st word as img or concat word+img on each time step


import pickle
class mappers(object):
    def __init__(self):
        config = Config()
        with open(config.preprocessedVQAMapsFile, 'rb') as f:
            data = pickle.load(f)
        
        self.mapWordToID = data['wordToIDmap']
        self.singleCountWords = data['singleCountWords']
        self.classToAnsMap = data['classToAnsMap']
        self.ansToClassMap = data['ansToClassMap']
        
    def getSingleCountWords(self):
        return self.singleCountWords
    
    def getMapWordToID(self):
        return self.mapWordToID
    