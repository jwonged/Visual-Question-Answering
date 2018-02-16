'''
Created on 12 Jan 2018

@author: jwong
'''
from gensim.models.keyedvectors import KeyedVectors

def main():
    binPath = '/media/jwong/Transcend/GoogleNews-vectors-negative300.bin'
    txtPath = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    model = KeyedVectors.load_word2vec_format(binPath, binary=True)
    model.save_word2vec_format(txtPath, binary=False)


if __name__ == '__main__':
    main()