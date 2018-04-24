'''
Created on 30 Mar 2018

@author: jwong
'''
import unittest
from AttentionModel.utils.Input_Processor import InputProcessor

class mockConfig():
    pass

class Test(unittest.TestCase):


    def setUp(self):
        config = mockConfig()
        self.processor = InputProcessor(config)


    def tearDown(self):
        pass

    def testPadQuestionIDsUnequalLens(self):
        toTest = [[1,3,5,8],[2,4]]
        ansQns = [[1,3,5,8],[2,4,0,0]]
        ansLens = [4,2]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
        
    def testPadQuestionIDsSameLens(self):
        toTest = [[1,3,5,8],[2,4,5,1]]
        ansQns = [[1,3,5,8],[2,4,5,1]]
        ansLens = [4,4]
        resQns, resLens = self.processor._padQuestionIDs(toTest, 0)
        self.assertEqual(ansQns, resQns)
        self.assertEqual(ansLens, resLens)
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()