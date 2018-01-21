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
    
    #'imagePerWord' or 'imageAsFirstWord'
    modelStruct = 'imageAfterLSTM'
    
    nOutClasses = 1000
    batch_size = 20
    imgVecSize = 1024
    wordVecSize = 300
        
    nTrainEpochs = 30
    nEpochsWithoutImprov = 3
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
    
    #other options currently manual
    #Sentence padding = pad to max len
    #Handling words not in embeddings = ignore (or give UNK vec)
    #LSTM input = 1st word as img or concat word+img on each time step

    
    
    