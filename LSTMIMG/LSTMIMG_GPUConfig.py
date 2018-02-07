'''
Created on 7 Feb 2018

@author: jwong
'''

class LSTMIMG_GPUConfig(object):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self):
        pass
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'imageAfterLSTM'
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    imgVecSize = 1024
    wordVecSize = 300
    
    nTrainEpochs = 30
    nEpochsWithoutImprov = 3
    dropoutVal = 0.5 #standard 0.5, 1.0 for none
    modelOptimizer = 'adam'
    lossRate = 0.00005 #0.001 for adam, 0.01 for gradDesc
    lossRateDecay = 1 #noDecay = 1; usually ~0.9
    max_gradient_norm = -1 #for clipping; usually 4-5; -1 for none
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    shuffle = True
    trainEmbeddings = True
    
    csvResults = 'results/Pred_LSTMIMG_0702lrTuning.csv'
    logFile = 'results/Res_LSTMIMG_0702lrTuning.csv'
    saveModelFile = 'results/LSTMIMG_0702lrTuning'
    
    '''Pickle file Contains:
            singleCountWords , wordToIDmap, 
            classToAnsMap, ansToClassMap
    '''
    preprocessedVQAMapsFile = 'resources/preprocessedVQAmaps.pkl' #
    
    #Downloaded embeddings and shortened embeddings
    pretrainedw2v = '../resources/GoogleNews-vectors-negative300.txt' #
    shortenedEmbeddingsWithUNKFile = '../resources/cutW2VEmbeddingsWithUNK.npz' #
    
    #Qn_ID --> 'qn'
    rawQnTrain = '../resources/processedOpenEnded_trainQns.json' #
    rawQnValTestFile = '../resources/preprocessedValTestQnsOpenEnded.json' #
    
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
    trainAnnotFile = '../resources/AllTrainAnnotResolvedList.json' #
    valAnnotFile = '../resources/AllValAnnotResolvedList.json' #
    testAnnotFile = '../resources/AllTestAnnotResolvedList.json' #
    
    #img_id --> img feature vec (1024)
    trainImgFile = '../resources/VQAImgFeatures_Train.json' #
    valImgFile = '../resources/VQAImgFeatures_Val.json' #
    testImgFile = '../resources/VQAImgFeatures_Test.json'
