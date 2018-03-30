'''
Created on 20 Dec 2017

@author: jwong
'''
import pickle
import datetime
import calendar

class Config(object):
    '''
    Config for BOWIMG
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
        
        #quick hyperparameters
        self.randomSeed = args.seed if args.seed else 42
        self.nTrainEpochs = 40
        self.nEpochsWithoutImprov = 6
        self.decayAfterEpoch = 20
        self.dropoutVal = 0.5 #standard 0.5, 1.0 for none
        self.modelOptimizer = 'adam'
        self.learningRate = 0.0001 
        self.learningRateDecay = 0.95 #noDecay= 1; usually ~0.9
        
        self.debugMode = False
        
        now = datetime.datetime.now()
        self.dateAppend = '{}{}{}-{}'.format(now.day, calendar.month_name[now.month][:3],
                                            now.hour, now.minute)
        
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'BOWIMG'
    imgModel = 'googlenet' #'vgg' or 'googlenet' #for pre-trained img model
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    wordVecSize = 300
    imgVecSize = 1024
    
    unkWord = '<UNK>'
    probSingleToUNK = 0.1
    usePretrainedEmbeddings = True
    shuffle = True
    trainEmbeddings = True
    
    
        