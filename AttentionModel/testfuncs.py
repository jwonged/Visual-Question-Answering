'''
Created on 9 Mar 2018

@author: jwong
'''

import shelve
import numpy as np
import pickle

def checkload():
    outputFile = '../resources/vggTrainConv5_3Features_shelf'
    imgData = shelve.open(outputFile, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
    vec = imgData[str(363942)]
    print(vec)
    print(np.asarray(vec).shape)
    imgData.close()

if __name__ == '__main__':
    checkload()