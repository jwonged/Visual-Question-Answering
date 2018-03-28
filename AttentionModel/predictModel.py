'''
Created on 13 Feb 2018

@author: jwong
'''
from Image_AttModel import ImageAttentionModel
from Qn_AttModel import QnAttentionModel
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
    
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)
        
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()
    testReader.destruct()

def runValTest(args):
    #Val set's split -- test
    print('Running Val Test')
    config = Attention_LapConfig(load=True, args=args)
    
    valTestReader = TestProcessor(qnFile=config.valTestQns, 
                               imgFile=config.valImgFile, 
                               config=config)
    
    #valTestReader = InputProcessor(config.testAnnotFile, 
    #                             config.rawQnValTestFile, 
    #                             config.valImgFile, 
    #                             config,
    #                             is_training=False)
    
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreQnImAttModel, 
                           config.restoreQnImAttModelPath)
    model.runTest(valTestReader, 'testResFile.json')
    model.destruct()
    valTestReader.destruct()

def predAnalysis(args):
    print('Running Val Test')
    predFile = 'Pred_QnAtt47.9.csv'
    config = Attention_GPUConfig(load=True, args=args)
    valTestReader = InputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    
    model = QnAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runPredict(valTestReader, predFile)
    model.destruct()
    valTestReader.destruct()
    
def internalValTest(args):
    import sys
    sys.path.insert(0, '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering')
    from vqaTools.vqa import VQA
    from vqaTools.vqaEval import VQAEval
    
    config = Attention_LapConfig(load=False, args=args)
    annFile = config.originalAnnotVal
    quesFile = config.valTestQns
    resFile = 'testResFile.json'
    
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate() 
    
    # print accuracies
    print "\n"
    print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
    print "Per Question Type Accuracy is the following:"
    for quesType in vqaEval.accuracy['perQuestionType']:
        print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
    print "\n"
    print "Per Answer Type Accuracy is the following:"
    for ansType in vqaEval.accuracy['perAnswerType']:
        print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
    print "\n"
    

def runVisualiseVal():
    print('Running Visuals')
    config = Attention_LapConfig(load=True, args=args)
    reader = InputProcessor(config.testAnnotFile,
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
    reader = InputProcessor(config.trainAnnotFile, 
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
    from InputProcessor import OnlineProcessor
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
    reader = InputProcessor(config.testAnnotFile,
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
    parser.add_argument('-a', '--action', choices=['otest', 'vtest', 'vis', 'solve',
                                                    'qn', 'visval', 'solveqn', 'mkres',
                                                    'eval', 'pred'], default='vis')
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
    elif args.action == 'qn':
        visQnImgAtt()
    elif args.action == 'visval':
        runVisualiseVal()
    elif args.action == 'solveqn':
        solveqn()
    elif args.action == 'mkres':
        runValTest(args)
    elif args.action == 'eval':
        internalValTest(args)
    elif args.action == 'pred':
        predAnalysis(args)
        
        