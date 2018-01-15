'''
Created on 15 Jan 2018

@author: jwong
'''
import numpy as np

def getPretrainedw2v(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["vectors"]