'''
Created on 9 Feb 2018

@author: jwong
'''
import pickle
import datetime
import calendar

class Config(object):
    '''
    Config file for the Attention Model
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
        self.nTrainEpochs = 60
        self.nEpochsWithoutImprov = 7
        self.decayAfterEpoch = 20
        self.dropoutVal = 0.5 #standard 0.5, 1.0 for none
        self.modelOptimizer = 'adam'
        self.learningRate = 0.0001
        self.learningRateDecay = 0.95 #noDecay= 1; usually ~0.9
        self.max_gradient_norm = 4 #for clipping; usually 4-5; -1 for none
        self.attentionType = args.att if args.att else 'qn'
        self.attentionFunc = args.attfunc if args.attfunc else 'softmax'
        self.attComb = args.attComb if args.attComb else 'concat' #or add or mult
        self.stackAtt = args.stackAtt
        self.qnAttf = args.qnAttf if args.qnAttf else 'softmax'
        self.mmAtt = args.mmAtt
        
        self.debugMode = args.debugmode
        self.notopk = args.notopk
        self.noqnatt = args.noqnatt
        
        now = datetime.datetime.now()
        self.dateAppend = '{}{}{}-{}'.format(now.day, calendar.month_name[now.month][:3],
                                            now.hour, now.minute)
        
    
    #'imagePerWord' or 'imageAsFirstWord' or 'imageAfterLSTM'
    modelStruct = 'imageAfterLSTM'
    imgModel = 'vgg' #'vgg' or 'googlenet' #for pre-trained img model
    
    nOutClasses = 3147  #1000 or >3 freq=3127 or 17140 (all)
    batch_size = 32
    wordVecSize = 300
    imgVecSize = [512, 14, 14]
    
    LSTM_num_units = 512
    fclayerAfterLSTM = 1024
    elMult = True #False = concat; True = Mult
    LSTMType = 'bi' #'bi' or 'single'
    '''
    nTrainEpochs = 50
    nEpochsWithoutImprov = 5
    decayAfterEpoch = 20
    dropoutVal = 0.5 #standard 0.5, 1.0 for none
    modelOptimizer = 'adam'
    learningRate = 0.0001 #0.001 for adam, 0.01 for gradDesc
    learningRateDecay = 0.95 #noDecay = 1; usually ~0.9
    max_gradient_norm = 4 #for clipping; usually 4-5; -1 for none
    '''
    unkWord = '<UNK>'
    probSingleToUNK = 0.05
    usePretrainedEmbeddings = False
    shuffle = True
    trainEmbeddings = True
    removeQnMark = False
    
    
        
