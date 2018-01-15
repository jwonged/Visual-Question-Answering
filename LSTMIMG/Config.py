'''
Created on 14 Jan 2018

@author: jwong
'''

class Config(object):
    '''
    Config file for LSTMIMG
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    self.pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    self.idQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/QnWordsIDlistTrain_OpenEnded.json'
    self.idValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/QnWordsIDlistValTest_OpenEnded.json'