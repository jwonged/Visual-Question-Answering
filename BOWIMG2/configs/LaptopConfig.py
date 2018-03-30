'''
Created on 14 Jan 2018

@author: jwong
'''
from Config import Config

class BOWIMG_LapConfig(Config):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load, args):
        super(BOWIMG_LapConfig, self).__init__(load, args)
        
        print('Using GoogLeNet config')
        #img_id --> img feature vec (1024)
        self.trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
        self.valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
        self.testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'

    csvResults = '/media/jwong/Transcend/VQAresults/ImgAtt/20MarAtt/Pred_Att2003.csv'
    logFile = '/media/jwong/Transcend/VQAresults/ImgAtt/20MarAtt/Res_Att2003.csv'
    
    saveModelFile = '/media/jwong/Transcend/VQADataset/DummySets/LSTMIMG-proto'
    saveModelPath = '/media/jwong/Transcend/VQADataset/DummySets/'
    restoreModel = '/media/jwong/Transcend/VQAresults/ImgAtt/20MarAtt/att20Mar-35.meta'
    restoreModelPath = '/media/jwong/Transcend/VQAresults/ImgAtt/20MarAtt/'
    testOfficialResultFile = '/media/jwong/Transcend/VQADataset/VQASubmissions/47LSTMIMG11MarSubmission.json'
    
    restoreQnImAttModel = '/media/jwong/Transcend/VQAresults/QnAndImgAtt/Att22Mar0-12/att22Mar0-12.meta'
    restoreQnImAttModelPath = '/media/jwong/Transcend/VQAresults/QnAndImgAtt/Att22Mar0-12/'
    #restoreModel = '/media/jwong/Transcend/VQADataset/DummySets/LSTMIMG-proto.meta'
    #saveModelPath = '/media/jwong/Transcend/VQADataset/DummySets/'
    
    trainImgFile = '/media/jwong/Transcend/conv5_3_shelves/vggTrainConv5_3Features_shelf'
    valImgFile = '/media/jwong/Transcend/conv5_3_shelves/vggValConv5_3Features_shelf'
    trainImgPaths = '/media/jwong/Transcend/VQADataset/TrainSet/trainImgPaths.txt'
    valImgPaths = '/media/jwong/Transcend/VQADataset/ValTestSet/valImgPaths.txt'
    
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
    trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
    valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
    testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
    
    #trainAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/TrainAnnotList.json'
    #valAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/ValAnnotList.json'
    #testAnnotFile = '/media/jwong/Transcend/VQADataset/LSTMIMGData/TestAnnotList.json'
    
    originalAnnotTrain = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
    originalAnnotVal = '/media/jwong/Transcend/VQADataset/ValTestSet/mscoco_val2014_annotations.json'
    
    