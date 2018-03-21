'''
Created on 13 Feb 2018

@author: jwong
'''
from Image_AttModel import ImageAttentionModel
from Attention_LapConfig import Attention_LapConfig
from Attention_GPUConfig import Attention_GPUConfig
from InputProcessor import InputProcessor, TestProcessor
from model_utils import OutputGenerator
import argparse


        
def loadOfficialTest(args):
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()
    testReader.destruct()

def runValTest(args):
    #Val set's split -- test
    print('Running Val Test')
    config = Attention_LapConfig(load=True, args=args)
    valTestReader = InputProcessor(config.testAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=False)
    
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runPredict(valTestReader)
    model.destruct()
    valTestReader.destruct()

def runVisualise():
    print('Running Visuals')
    config = Attention_LapConfig(load=True, args=args)
    reader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=False)
    
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    alphas, img_ids, qns, preds = model.runPredict(reader)
    model.destruct()
    reader.destruct()
    
    imgpaths = '/media/jwong/Transcend/VQADataset/TrainSet/trainImgPaths.txt'
    out = OutputGenerator(imgpaths)
    out.displayOutput(alphas, img_ids, qns, preds)

from PIL import Image                  
def solve():
    print('Running solve')
    imgpaths = '/media/jwong/Transcend/VQADataset/TrainSet/trainImgPaths.txt'
    config = Attention_LapConfig(load=True, args=args)
    out = OutputGenerator(imgpaths)
    img_id = raw_input('Img_id--> ')
    img = Image.open(out.convertIDtoPath(img_id))
    img.show()
    
    qn = raw_input('Question--> ')
    print(qn)
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath) 
    alpha, pred = model.solve(qn, img_id)
    out.displaySingleOutput(alpha, img_id, qn, pred)
    
    '''
    262415
    148639
    47639
    235328
    118572
    246470
    499024
    '''

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Display all print statement', 
                        action='store_true')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--model', choices=['qn', 'im'], default='qn')
    parser.add_argument('-a', '--action', choices=['otest', 'vtest', 'vis', 'solve'], default='vis')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    if args.action == 'otest':
        print 'otest'
        loadOfficialTest(args)
    elif args.action == 'vis':
        runVisualise()
    elif args.action == 'solve':
        solve()
    