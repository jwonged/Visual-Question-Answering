'''
Created on 9 Feb 2018

@author: jwong
'''
import pickle
import datetime
import calendar

class Config(object):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load, args):
        if load:
            print('Reading ' + self.preprocessedVQAMapsFile)
            with open(self.preprocessedVQAMapsFile, 'rb') as f:
                data = pickle.load(f)
            
            self.mapAnsToClass = data['ansToClassMap']
            print('Using {} answer classes'.format(len(self.mapAnsToClass)))
            
            self.classToAnsMap = data['classToAnsMap']
            self.classToAnsMap[-1] = -1
            
            self.mapWordToID = data['wordToIDmap']
            self.singleCountWords = data['singleCountWords']
            self.vocabSize = len(self.mapWordToID)
        
        self.randomSeed = args.seed if args.seed else 42
        self.elMult = args.useconcat #False = concat; True = Mult
        self.LSTMType = args.lstmtype #'bi' or 'single'
        self.usePretrainedEmbeddings = args.useuntrainedembed
        self.trainEmbeddings = args.donttrainembed
        self.shuffle = args.noshuffle
        self.mmAtt = args.mmAtt
        
        self.debugMode = False
        
        now = datetime.datetime.now()
        self.dateAppend = '{}{}{}-{}'.format(now.day, calendar.month_name[now.month][:3],
                                            now.hour, now.minute)
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'imageAfterLSTM'
    imgModel = 'googlenet' #'vgg' or 'googlenet' #for pre-trained img model
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    wordVecSize = 300
    
    LSTM_num_units = 512
    fclayerAfterLSTM = 1024
    
    LSTMCellSizes = [512] #for single only
    
    nTrainEpochs = 50
    nEpochsWithoutImprov = 5
    dropoutVal = 0.5 #standard 0.5, 1.0 for none
    modelOptimizer = 'adam'
    decayAfterEpoch = 15
    learningRate = 0.0001 #0.001 for adam, 0.01 for gradDesc
    learningRateDecay = 0.95 #noDecay = 1; usually ~0.9
    max_gradient_norm = 4 #for clipping; usually 4-5; -1 for none
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    
    
    if imgModel == 'googlenet':
        imgVecSize = 1024
    else:
        imgVecSize = 4096
        
class Config_46GoogLeNet(object):
    '''
    Config file for LSTMIMG
    '''
    def __init__(self, load):
        if load:
            print('Reading ' + self.preprocessedVQAMapsFile)
            with open(self.preprocessedVQAMapsFile, 'rb') as f:
                data = pickle.load(f)
            
            self.mapAnsToClass = data['ansToClassMap']
            print('Using {} answer classes'.format(len(self.mapAnsToClass)))
            
            self.classToAnsMap = data['classToAnsMap']
            self.classToAnsMap[-1] = -1
            
            self.mapWordToID = data['wordToIDmap']
            self.singleCountWords = data['singleCountWords']
            self.vocabSize = len(self.mapWordToID)
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'imageAfterLSTM'
    imgModel = 'vgg' #'vgg' or 'googlenet' #for pre-trained img model
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    wordVecSize = 300
    
    LSTM_num_units = 512
    elMult = True #False = concat; True = Mult
    LSTMType = 'bi' #'bi' or 'single'
    LSTMCellSizes = [512] #for single only
    
    nTrainEpochs = 50
    nEpochsWithoutImprov = 5
    dropoutVal = 0.5 #standard 0.5, 1.0 for none
    modelOptimizer = 'adam'
    learningRate = 0.0001 #0.001 for adam, 0.01 for gradDesc
    learningRateDecay = 1 #noDecay = 1; usually ~0.9
    max_gradient_norm = -1 #for clipping; usually 4-5; -1 for none
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    usePretrainedEmbeddings = True
    shuffle = True
    trainEmbeddings = True
    
    if imgModel == 'googlenet':
        imgVecSize = 1024
    else:
        imgVecSize = 4096