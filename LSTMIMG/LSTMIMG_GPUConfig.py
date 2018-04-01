'''
Created on 7 Feb 2018

@author: jwong
'''
from Config import Config

class LSTMIMG_GPUConfig(Config):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load, args):
        super(LSTMIMG_GPUConfig, self).__init__(load, args)
        
        if self.imgModel == 'googlenet':
            print('Using GoogLeNet config')
            self.trainImgFile = '../resources/VQAImgFeatures_Train.json' #
            self.valImgFile = '../resources/VQAImgFeatures_Val.json' #
            self.testImgFile = '../resources/VQAImgFeatures_Test.json'
        else:
            print('Using VGGNet config')
            self.trainImgFile = '../resources/vggTrainImgFeatures.json'
            self.valImgFile = '../resources/vggValImgFeatures.json'
            self.testOfficialImgFile = '../resources/vggTestOfficialImgFeatures.json'
            
        self.saveModelPath = './results/LSTM{}/'.format(self.dateAppend) 
        self.saveModelFile = self.saveModelPath + 'LSTM' + self.dateAppend
        
        self.csvResults = self.saveModelPath + 'Pred_LSTM_{}.csv'.format(self.dateAppend)
        self.logFile = self.saveModelPath + 'Res_LSTM_{}.csv'.format(self.dateAppend)
        
        self.restoreModel = args.restorefile if args.restorefile else '.meta'
        self.restoreModelPath = args.restorepath if args.restorepath else './'
        
        self.testOfficialResultFile = self.restoreModelPath + 'LSTM{}Submission.json'.format(self.dateAppend)
        
    
    #csvResults = 'results/Pred_LSTMIMG_10Mar.csv'
    #logFile = 'results/Res_LSTMIMG_10Mar.csv'
    #saveModelFile = 'results/LSTMIMG1003'
    
    testOfficialDevQns = '../resources/OpenEnded_mscoco_test-dev2015_questions.json'
    testOfficialImgFeatures = '../resources/OfficialTestGoogLeNetImgFeats.json'
    
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
    
    originalValQns = '../resources/OpenEnded_mscoco_val2014_questions.json'
    '''
    Annotations file in list with resolved single answer
    [
        {
            "question_id" : int,
            "image_id" : int,
            "question_type" : str,
            "answer_type" : str,
            "answers" : [answer{
                        "answer_id" : int,
                        "answer" : str,
                        "answer_confidence": str,
                        }]
            "multiple_choice_answer" : str
        }, ...
    ]
    248,349 train
    60,753 val
    60,756 test
    '''
    trainAnnotFile = '../resources/AllTrainAnnotResolvedList.json'
    valAnnotFile = '../resources/AllValAnnotResolvedList.json'
    testAnnotFile = '../resources/AllTestAnnotResolvedList.json'
    
    trainAnnotFileUnresolved = '../resources/TrainAnnotList.json'
    valAnnotFileUnresolved = '../resources/ValAnnotList.json'
    testAnnotFileUnresolved = '../resources/TestAnnotList.json'
    
    
    #raw image files
    #http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    #http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    #http://msvocds.blob.core.windows.net/coco2015/test2015.zip
    
