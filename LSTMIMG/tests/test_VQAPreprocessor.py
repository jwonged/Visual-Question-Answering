'''
Created on 23 Jan 2018

@author: jwong
'''
import unittest
from LSTMIMG.VQAPreprocessor import VQAPreprocessor
from LSTMIMG.Config import Config


class Test(unittest.TestCase):


    def setUp(self):
        self.processor = VQAPreprocessor(Config())


    def tearDown(self):
        pass


    def test_getWordFreqsFromQnFile(self):
        rawQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    
        self.processor.getVocabForEmbeddings()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_getWordFreqsFromQnFile']
    unittest.main()