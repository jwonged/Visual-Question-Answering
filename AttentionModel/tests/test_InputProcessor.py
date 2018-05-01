'''
Created on 30 Mar 2018

@author: jwong
'''
import unittest
from AttentionModel.utils.Input_Processor import InputProcessor
from AttentionModel.utils.TrainProcessors import  TestProcessor

class mockConfig():
    pass

class Test(unittest.TestCase):


    def setUp(self):
        config = mockConfig()
        self.processor = InputProcessor(config)
        

    def testPadQuestionIDsUnequalLens(self):
        toTest = [[1,3,5,8],[2,4]]
        ansQns = [[1,3,5,8],[2,4,0,0]]
        ansLens = [4,2]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
        
    def testPadQuestionIDsSameLens(self):
        toTest = [[2,4,5,1], [1,3,5,8]]
        ansQns = [[2,4,5,1], [1,3,5,8]]
        ansLens = [4,4]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
    
    def testPadQuestionIDsBlank(self):
        toTest = []
        with self.assertRaises(ValueError):
            self.processor._padQuestionIDs(toTest, 0)
    
    def testPadQuestionIDsUnequalLenMultiple3(self):
        toTest = [[1,3,5,8],[2,4], [2,4, 7, 9]]
        ansQns = [[1,3,5,8],[2,4,0,0], [2,4,7,9]]
        ansLens = [4,2,4]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
    
    def testPadQuestionIDsUnequalLenMultipleUneven(self):
        toTest = [[1,3,5,8],[2], [2,4, 7]]
        ansQns = [[1,3,5,8],[2,0,0,0], [2,4,7,0]]
        ansLens = [4,1,3]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()