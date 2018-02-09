'''
Created on 9 Feb 2018

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
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    imgVecSize = 1024
    wordVecSize = 300
    LSTM_num_units = 512
    elMult = True #False = concat; True = Mult
    LSTMType = 'bi' #bi or single
    #LSTMCellSizes = [128, 256] #for single only
    
    nTrainEpochs = 30
    nEpochsWithoutImprov = 5
    dropoutVal = 0.5 #standard 0.5, 1.0 for none
    modelOptimizer = 'adam'
    lossRate = 0.0001 #0.001 for adam, 0.01 for gradDesc
    lossRateDecay = 1 #noDecay = 1; usually ~0.9
    max_gradient_norm = -1 #for clipping; usually 4-5; -1 for none
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    shuffle = True
    trainEmbeddings = True