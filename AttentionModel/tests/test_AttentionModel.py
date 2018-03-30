'''
Created on 12 Mar 2018

@author: jwong
'''
import unittest
import argparse

from AttentionModel.configs import Attention_LapConfig
from AttentionModel.model.Qn_AttModel import QnAttentionModel

class Test(unittest.TestCase):

    def test_AttentionModel(self):
        pass
    
    def test_QnAttentionModel_construct(self):
        args = mockArgs()
        config = Attention_LapConfig(load=True, args=args)
        model = QnAttentionModel(config)
        model.construct()

class mockArgs():
    seed = 1
    att = 'qn'
    restorefile = ''
    restorepath = ''
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()