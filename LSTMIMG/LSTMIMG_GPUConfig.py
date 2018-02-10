'''
Created on 7 Feb 2018

@author: jwong
'''
from Config import Config

class LSTMIMG_GPUConfig(Config):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load):
        super(LSTMIMG_GPUConfig, self).__init__(load)
    
    csvResults = 'results/Pred_LSTMIMG_1002_singleLSTMlr.csv'
    logFile = 'results/Res_LSTMIMG_1002_singleLSTMlr.csv'
    saveModelFile = 'results/LSTMIMG_0802lrTuning'
    
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
