'''
Created on 12 Mar 2018

@author: jwong
'''
import unittest
import argparse
import tensorflow as tf
import numpy as np

from AttentionModel.model.Qn_AttModel import QnAttentionModel

class AttModelTest(tf.test.TestCase):
    
    def sigmoid(self, x):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] =  1/(1+np.exp(-x[i][j]))
        return x
        
    
    def testComputeSigmoid_FullSeq(self):
        with self.test_session():
            toTest = [[0.7310586, 0.7310586], [0.880797,  0.880797 ]]
            
            correctVal = self.sigmoid(toTest)#tf.nn.sigmoid(toTest) 
            #print(correctVal)
            res = self._computeSigmoid(toTest, [2,2])
            #print(res.eval())
            self.assertAllEqual(correctVal,res)
    
    def testComputeSigmoid_unevenSeq(self):
        with self.test_session():
            toTest = [[0.66, 0.731586], [0.3797,  0.17 ]]
            
            correctVal = self.sigmoid(toTest)#tf.nn.sigmoid(toTest) 
            #print(correctVal)
            res = self._computeSigmoid(toTest, [2,1])
            #print(res.eval())
            self.assertAllEqual(correctVal,res)
    
    def testComputeSigmoid_overshotSeq(self):
        with self.test_session():
            toTest = [[0.66, 0.7369], [0.4,  0.8 ]]
            
            correctVal = self.sigmoid(toTest)#tf.nn.sigmoid(toTest) 
            #print(correctVal)
            res = self._computeSigmoid(toTest, [2,5])
            #print(res.eval())
            self.assertAllEqual(correctVal,res)
            
    def testComputeSigmoid_bothReducedSeq(self):
        with self.test_session():
            toTest = [[0.4, 0.269,0.06], [0.1,  0.9,0.04 ]]
            
            correctVal = self.sigmoid(toTest)#tf.nn.sigmoid(toTest) 
            #print(correctVal)
            res = self._computeSigmoid(toTest, [2,1])
            #print(res.eval())
            self.assertAllEqual(correctVal,res)


class mockArgs():
    seed = 1
    att = 'qn'
    restorefile = ''
    restorepath = ''
    
if __name__ == "__main__":
    tf.test.main()
