'''
Created on 15 Jan 2018

@author: jwong
'''

from Base_Model import BaseModel
from utils.model_utils import getPretrainedw2v
import numpy as np
import tensorflow as tf 

class BOWIMGModel(BaseModel):
    '''
    VQA Model implementing BOW+pre-trained CNN
    '''
    def __init__(self, config):
        super(BOWIMGModel, self).__init__(config)
    
    def _addPlaceholders(self):
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        
        # shape = (batch size, length of image feature vector)
        self.img_vecs = tf.placeholder(tf.float32, 
                                       shape=[None, self.config.imgVecSize], 
                                       name="img_vecs")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, cancel(max length of sentence in batch))
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    
    def comment(self):
        return 'Standard BOWIMG baseline'
    
    def _addEmbeddings(self):
        #add word embeddings
        with tf.variable_scope("words"):
            if self.config.usePretrainedEmbeddings:
                print('Using pretrained w2v embeddings')
                pretrainedEmbeddings = getPretrainedw2v(self.config.shortenedEmbeddingsWithUNKFile)
                wordEmbedsVar = tf.Variable(pretrainedEmbeddings,
                        name="wordEmbedsVar",
                        dtype=tf.float32,
                        trainable=self.config.trainEmbeddings)
            else:
                print('Using untrained embeddings')
                wordEmbedsVar = tf.get_variable(
                        name='_word_embeddings',
                        shape=[self.config.vocabSize, self.config.wordVecSize], 
                        dtype=tf.float32)
                
        #embedding matrix, word_ids
        word_embeddings = tf.nn.embedding_lookup(wordEmbedsVar,
                self.word_ids, name="word_embeddings")
        
        word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
        return word_embeddings #[b, maxLen, 300]
    
    def construct(self):
        self._addPlaceholders()
        print('ImgVecSize: {}'.format(self.img_vecs.get_shape()))
        
        word_embeddings = self._addEmbeddings() #[b x maxlen x 300]
        
        #mask paddings and sum words
        mask = tf.sequence_mask(self.sequence_lengths) #[b x seqLens]
        float_mask =  tf.expand_dims(tf.to_float(mask), axis=-1) #[b x seqLens x 1]
        masked_embeddings = tf.multiply(word_embeddings, float_mask) #[bxmaxlenx300]
        bows = tf.reduce_sum(masked_embeddings, axis=1) #[bx300]
        
        multimodalOutput = tf.concat([bows, self.img_vecs], axis=-1) #[bx1324]
        
        #for debugging
        if self.config.debugMode:
            print('Shape of float_mask: {}'.format(float_mask.get_shape()))
            print('Shape of masked_embeddings: {}'.format(masked_embeddings.get_shape()))
            print('Shape of bows: {}'.format(bows.get_shape()))
            self.word_embeddings = word_embeddings
            self.float_mask = float_mask
            self.mask_embeddings = masked_embeddings
            self.bows = bows
            self.multimodalOutput = multimodalOutput
        
        #fully connected layer
        with tf.variable_scope("proj"):
            y = self._denseLayer(multimodalOutput, 
                                 self.config.wordVecSize+self.config.imgVecSize, 
                                 self.config.nOutClasses, activationf=False)
            print('Shape of y: {}'.format(y.get_shape()))
            
        #predict & get accuracy
        self.labels_pred = tf.cast(tf.argmax(tf.nn.softmax(y), axis=1), tf.int32, name='labels_pred')
        
        is_correct_prediction = tf.equal(self.labels_pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
        
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        self.loss = tf.reduce_mean(crossEntropyLoss)
        
        self.predProbs = tf.nn.softmax(y, name='softmax')
        self.predscore = (tf.argmax(self.predProbs, axis=1))
        self.topK = tf.nn.top_k(self.predProbs, k=5, name='topK')

        # Add to tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    def _denseLayer(self, x, nInputs, nOutputs, activationf=True):
        w = tf.Variable(tf.zeros([nInputs, nOutputs]))
        b = tf.Variable(tf.zeros([nOutputs]))
        y = tf.matmul(x, w) + b
        
        if activationf:
            y = tf.tanh(y)
        
        return y
        
    