'''
Created on 14 Jan 2018

@author: jwong
'''
from Config import Config

class LSTMIMG_LapConfig(Config):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load, args):
        super(LSTMIMG_LapConfig, self).__init__(load, args)
        
        if self.imgModel == 'googlenet':
            print('Using GoogLeNet config')
            #img_id --> img feature vec (1024)
            self.trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
            self.valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
            self.testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
        else:
            print('Using VGGNet config')
            #img_id --> img feature vec (4096)
            self.trainImgFile = '../resources/vggTrainImgFeatures.json'
            self.valImgFile = '../resources/vggValImgFeatures.json'
            self.testImgFile = '../resources/vggTestOfficialImgFeatures.json'
    
    csvResults = '/media/jwong/Transcend/VQAresults/LSTMIMG/Pred_LSTMIMG1402.csv'
    logFile = '/media/jwong/Transcend/VQAresults/LSTMIMG/Res_LSTMIMG1402.csv'
    saveModelFile = '/media/jwong/Transcend/VQADataset/DummySets/LSTMIMG-proto'
    saveModelPath = '/media/jwong/Transcend/VQADataset/DummySets/'
    restoreModel = '/media/jwong/Transcend/VQAresults/14FebTest/LSTMIMG_46Config.meta'
    saveModelPath = '/media/jwong/Transcend/VQAresults/14FebTest/'
    testOfficialResultFile = '/media/jwong/Transcend/VQADataset/VQASubmissions/LSTMIMG1402Submission.json'
    
    #restoreModel = '/media/jwong/Transcend/VQADataset/DummySets/LSTMIMG-proto.meta'
    #saveModelPath = '/media/jwong/Transcend/VQADataset/DummySets/'
    
    
    
    '''Pickle file Contains:
            singleCountWords , wordToIDmap, 
            classToAnsMap, ansToClassMap
    '''
    
    testOfficialQns = '/media/jwong/Transcend/VQADataset/OfficialTestSet/Questions_Test_mscoco/OpenEnded_mscoco_test2015_questions.json'
    testOfficialDevQns = '/media/jwong/Transcend/VQADataset/OfficialTestSet/Questions_Test_mscoco/OpenEnded_mscoco_test-dev2015_questions.json'
    testOfficialImgFeatures = '/media/jwong/Transcend/VQADataset/OfficialTestSet/test2015/officialTestImgFeatures.json'
    
    preprocessedVQAMapsFile = '/media/jwong/Transcend/VQADataset/preprocessedVQAmaps.pkl'
    
    valTestQns = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json'
    
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    shortenedEmbeddingsWithUNKFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddingsWithUNK.npz'
    
    #empty for now
    datasetVocabWithUNK = '/media/jwong/Transcend/VQADataset/FullVQAVocabWwithUNK.txt'
    
    #Qn_ID --> 'qn'
    rawQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    rawQnValTestFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
    
    
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
    #trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
    #valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
    #testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
    
    trainAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/TrainAnnotList.json'
    valAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/ValAnnotList.json'
    testAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/TestAnnotList.json'
    
    originalAnnotTrain = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
    originalAnnotVal = '/media/jwong/Transcend/VQADataset/ValTestSet/mscoco_val2014_annotations.json'
    
    
    #######Deprecated files##########
    #contains vocab from train/val/test questions and answers
    vocabFile = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    fullDatasetVocab2301 = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    shortenedEmbeddingsWithoutUNKFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddings.npz'
    
    ansClass1000File = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
    mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/allTrainAnswers.csv'
    
    #Qn_ID --> id, id, id...
    idQnTrainFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/QnWordsIDlistTrain_OpenEnded.json'
    idValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/QnWordsIDlistValTest_OpenEnded.json'
    

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
    