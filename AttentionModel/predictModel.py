'''
Created on 13 Feb 2018

@author: jwong
'''
from model.Image_AttModel import ImageAttentionModel
from model.Qn_AttModel import QnAttentionModel
from configs.Attention_LapConfig import Attention_LapConfig
from configs.Attention_GPUConfig import Attention_GPUConfig
from utils.TrainProcessors import AttModelInputProcessor, TestProcessor
from utils.Output_Generator import OutputGenerator
from utils.Online_Processor import OnlineProcessor
import argparse

'''
1) Visualise qn att
2) Solve
3) Visualise multiple img att
'''

def runVisualiseVal():
    print('Running Visuals')
    config = Attention_LapConfig(load=True, args=args)
    reader = AttModelInputProcessor(config.testAnnotFile,
                                 config.rawQnValTestFile,
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    alphas, img_ids, qns, preds = model.runPredict(
        reader, config.csvResults, 9, mini=True)
    model.destruct()
    reader.destruct()
    
    out = OutputGenerator(config.valImgPaths)
    out.displayOutput(alphas, img_ids, qns, preds)

def runVisualise():
    print('Running Visuals')
    config = Attention_LapConfig(load=True, args=args)
    reader = AttModelInputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=False)
    
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    alphas, img_ids, qns, preds = model.runPredict(
        reader, config.csvResults, 5, mini=True)
    model.destruct()
    reader.destruct()
    
    out = OutputGenerator(config.trainImgPaths)
    out.displayOutput(alphas, img_ids, qns, preds)

from PIL import Image                  
def solve():
    print('Running solve')
    config = Attention_LapConfig(load=True, args=args)
    out = OutputGenerator(config.trainImgPaths)
    #img_id = raw_input('Img_id--> ')
    img_id = str(262415)
    img = Image.open(out.convertIDtoPath(str(img_id)))
    img.show()
    
    qn = raw_input('Question--> ')
    print(qn)
    model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath) 
    alpha, pred = model.solve(qn, img_id)
    out.displaySingleOutput(alpha, img_id, qn, pred)
    
    ''' -a otest -r ./results/Att21Mar1334/att21Mar1334.meta -p ./results/Att21Mar1334/
    -a otest -r ./results/Att22Mar0-12/att22Mar0-12.meta -p ./results/Att22Mar0-12/
    262415
    148639
    47639
    235328
    118572
    246470
    499024
    '''
    
def solveqn():
    print('Running solve')
    config = Attention_LapConfig(load=True, args=args)
    out = OutputGenerator(config.valImgPaths)
    #img_id = raw_input('Img_id--> ')
    img_id = str(337826) #214587
    img = Image.open(out.convertIDtoPath(str(img_id)))
    img.show()
    model = QnAttentionModel(config)
    model.loadTrainedModel(config.restoreQnImAttModel, 
                           config.restoreQnImAttModelPath)
    processor = OnlineProcessor(config.valImgFile, config)
    
    for i in range(5):
        qn = raw_input('Question--> ')
        print(qn)
        qnalpha, alpha, pred = model.solve(qn, img_id, processor)
        out.displaySingleOutput(alpha, img_id, qn, pred)


def visQnImgAtt():
    print('Running qn Visuals')
    config = Attention_LapConfig(load=True, args=args)
    reader = AttModelInputProcessor(config.testAnnotFile,
                                 config.rawQnValTestFile,
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    model = QnAttentionModel(config)
    model.loadTrainedModel(config.restoreQnImAttModel, 
                           config.restoreQnImAttModelPath)
    qnAlphas, alphas, img_ids, qns, preds = model.runPredict(
        reader, config.csvResults, 7, mini=True)
    model.destruct()
    
    out = OutputGenerator(config.valImgPaths)
    out.displayQnImgAttention(qnAlphas, alphas, img_ids, qns, preds)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Display all print statement', 
                        action='store_true')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('--attfunc', choices=['sigmoid', 'softmax'], default='softmax')
    parser.add_argument('-a', '--action', choices=['vtest', 'vis', 'solve',
                                                    'qn', 'visval', 'solveqn', 'pred'], default='vis')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    if args.action == 'vis':
        runVisualise()
    elif args.action == 'solve':
        solve()
    elif args.action == 'qn':
        visQnImgAtt()
    elif args.action == 'visval':
        runVisualiseVal()
    elif args.action == 'solveqn':
        solveqn()
        
        