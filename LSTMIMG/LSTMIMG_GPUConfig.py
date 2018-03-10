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
    
    csvResults = 'results/Pred_LSTMIMG_10Mar.csv'
    logFile = 'results/Res_LSTMIMG_10Mar.csv'
    saveModelFile = 'results/LSTMIMG1003'
    
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
    trainAnnotFile = '../resources/TrainAnnotList.json'
    valAnnotFile = '../resources/ValAnnotList.json'
    testAnnotFile = '../resources/TestAnnotList.json'
    
    
    #raw image files
    #http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    #http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    #http://msvocds.blob.core.windows.net/coco2015/test2015.zip
    
