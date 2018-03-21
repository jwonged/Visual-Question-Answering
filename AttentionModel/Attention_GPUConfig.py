'''
Created on 7 Feb 2018

@author: jwong
'''
from Config import Config


class Attention_GPUConfig(Config):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load, args):
        super(Attention_GPUConfig, self).__init__(load, args)
        
        self.saveModelPath = './results/Att{}/'.format(self.dateAppend) 
        self.saveModelFile = self.saveModelPath + 'att' + self.dateAppend
        
        self.csvResults = self.saveModelPath + 'Pred_att_{}.csv'.format(self.dateAppend)
        self.logFile = self.saveModelPath + 'Res_att_{}.csv'.format(self.dateAppend)
        self.testOfficialResultFile = 'results/att{}Submission.json'.format(self.dateAppend)
        
        self.restoreModel = args.restorefile if args.restorefile else '.meta'
        self.restoreModelPath = args.restorepath if args.restorepath else './'
        
    trainImgFile = '../resources/vggTrainConv5_3Features_shelf'
    valImgFile = '../resources/vggValConv5_3Features_shelf' #also internal test
    
    testOfficialDevQns = '../resources/OpenEnded_mscoco_test-dev2015_questions.json'
    testOfficialImgFeatures = '../resources/vggTestOfficialconv5_3Features_shelf'
    
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
    #trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/'
    #valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/'
    #testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/'
    
    trainAnnotFile = '../resources/AllTrainAnnotResolvedList.json'
    valAnnotFile = '../resources/AllValAnnotResolvedList.json'
    testAnnotFile = '../resources/AllTestAnnotResolvedList.json'
    
    #trainAnnotFile = '../resources/TrainAnnotList.json'
    #valAnnotFile = '../resources/ValAnnotList.json'
    #testAnnotFile = '../resources/TestAnnotList.json'
    
    
    #raw image files
    #http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    #http://msvocds.blob.core.windows.net/coco2014/val2014.zip
    #http://msvocds.blob.core.windows.net/coco2015/test2015.zip
    
