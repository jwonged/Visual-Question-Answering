'''
Created on 15 Jan 2018

@author: jwong
'''
from AttentionModel import AttentionModel
from Attention_LapConfig import Attention_LapConfig
from Attention_GPUConfig import Attention_GPUConfig
from InputProcessor import InputProcessor

def runtrain():
    #config = Attention_LapConfig(load=True)
    config = Attention_GPUConfig(load=True)
    
    trainReader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=True)
    
    valReader = InputProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    model = AttentionModel(config)
    model.construct()
    model.train(trainReader, valReader)
    model.destruct()
    trainReader.destruct()
    valReader.destruct()
        
if __name__ == '__main__':
    runtrain()