'''
Created on 15 Jan 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from Config import Config

def runtrain():
    model = LSTMIMGmodel(Config)
    model.construct()
    
    
    

if __name__ == '__main__':
    runtrain()