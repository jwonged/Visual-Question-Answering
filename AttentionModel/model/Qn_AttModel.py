'''
Created on 20 Mar 2018

@author: jwong
'''

import json
import os
import pickle
import time
import csv

from utils.model_utils import getPretrainedw2v
from model.Base_AttModel import BaseModel
import numpy as np
import tensorflow as tf 


class QnAttentionModel(BaseModel):
    '''
    VQA Model implementing attention over question and images
    '''
    def __init__(self, config):
        super(QnAttentionModel, self).__init__(config)
    
    def comment(self):
        return 'Better masking on QnAtt model with img att layer and qn boolean masking'
    
    def _addPlaceholders(self):
        # add network placeholders
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        
        # shape = (batch size, img tensor dimensions)
        self.img_vecs = tf.placeholder(tf.float32, 
                                       shape=[None, 512, 14, 14], 
                                       name="img_vecs")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, cancel(max length of sentence in batch))
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
    
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
        self.word_embeddings = tf.nn.embedding_lookup(wordEmbedsVar,
                self.word_ids, name="word_embeddings")
        
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)
        word_embeddings = self.word_embeddings
        return word_embeddings
    
    def _addLSTM(self, LSTMinput):
        #LSTM part
        with tf.variable_scope("lstm"):
            if self.config.LSTMType == 'bi':
                print('Using bi-LSTM')
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                
                #Out [batch_size, max_time, cell_output_size] output, outputState.c and .h
                (output_fw, output_bw), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, 
                    LSTMinput, 
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                #print('Shape of state.c: {}'.format(fw_state.c.get_shape()))
                print('Shape of output_fw: {}'.format(output_fw.get_shape()))
                print('Shape of output_fw: {}'.format(tf.shape(output_fw)))
                
                #[batch_size, max_time, cell.output_size]
                lstmOutput = tf.concat([output_fw, output_bw], axis=-1)
                
                '''
                #lstmOutput shape = LSTM_num_units * 4
                fw_out = tf.concat([fw_state.c, fw_state.h], axis=-1)
                bw_out = tf.concat([bw_state.c, bw_state.h], axis=-1)
                
                lstmOutput = tf.concat([fw_out, bw_out], axis=-1)
                print('Shape of LSTM output after concat: {}'.format(lstmOutput.get_shape()))
                
                #lstm output 2048 --> 1024
                lstmOutput = tf.layers.dense(inputs=lstmOutput,
                                           units=self.config.fclayerAfterLSTM,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                
                #potentially add dropout here
                '''
            else:
                print('Using Uni-LSTM')
                #rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in self.config.LSTMCellSizes]
                #multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                lstm_cell = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                _, lstmOutState = tf.nn.dynamic_rnn(cell=lstm_cell, 
                                                  inputs=self.word_embeddings, 
                                                  sequence_length=self.sequence_lengths, 
                                                  initial_state=None, 
                                                  dtype=tf.float32)
                lstmOutput = lstmOutState.c #output state 512
                #lstmOutput =  tf.concat([lstmOutState.c, lstmOutState.h], axis=-1) #1024
        
        lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
        return lstmOutput
    
    def _addQuestionAttention(self, lstmOutput):
        #########Question Attention##########
        with tf.variable_scope("qn_attention"):
            #qnAtt_f output: [b x seqLen x 1024]
            qnAtt_f =  tf.layers.dense(lstmOutput, units=lstmOutput.get_shape()[-1],
                                activation=tf.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()) 
            print('qnAtt_f shape: {}'.format(qnAtt_f.get_shape()))
            qnAtt_flat = tf.reshape(qnAtt_f, shape=[-1, qnAtt_f.get_shape()[-1]]) #[b*seqlen, 1024]
            qnAtt_beta = tf.get_variable("beta", shape=[qnAtt_f.get_shape()[-1], 1], dtype=tf.float32)
            
            qnAtt_flatWeights = tf.matmul(qnAtt_flat, qnAtt_beta) #[b*seqLen, 1]
            #qnAtt_flatWeights = tf.layers.dense(qnAtt_flat, units=1, activation=None)
            qnAtt_regionWeights = tf.reshape(
                qnAtt_flatWeights, shape=[-1, tf.shape(lstmOutput)[1]])
            #[b, seqLen(==nRegions)]
            
            #Mask output padding for softmax -- Take exp; mask; normalize
            exp_regionWs = tf.exp(qnAtt_regionWeights) #[b, maxLen]
            mask = tf.to_float(tf.sequence_mask(self.sequence_lengths)) #[b, maxLen]
            masked_expRegionWs = tf.multiply(exp_regionWs, mask) #[b, maxLen]
            denominator = tf.expand_dims(tf.reduce_sum(masked_expRegionWs, axis=-1), axis=-1) #[b, 1]
            self.qnAtt_alpha = tf.div(masked_expRegionWs, denominator, name='qn_alpha') #[b, maxLen]
            
            #self.qnAtt_alpha = tf.nn.softmax(qnAtt_regionWeights, name = 'qn_alpha')
            qnAtt_alpha = tf.expand_dims(self.qnAtt_alpha, axis=-1) #[b, seqLen, 1]
            qnContext = tf.reduce_sum(tf.multiply(qnAtt_alpha, lstmOutput), axis=1)
            #[b, 1024]
            
            if self.config.debugMode:
                self.qnAtt_regionWeights = qnAtt_regionWeights
                self.exp_regionWs = exp_regionWs
                self.mask = mask 
                self.masked_expRegionWs = masked_expRegionWs
                self.denominator = denominator
                self.qadim = qnAtt_alpha
            
            print('mask shape: {}'.format(mask.get_shape()))
            print('masked_regionWeights shape: {}'.format(masked_expRegionWs.get_shape()))
            print('self.qnAtt_alpha shape: {}'.format(self.qnAtt_alpha.get_shape()))
            
        return qnContext
    
    def _addImageAttention(self, qnContext, flattenedImgVecs):
        #########Image Attention layer##########
        with tf.variable_scope("image_attention"):
            qnContext_in = tf.layers.dense(inputs=qnContext,
                                           units=qnContext.get_shape()[-1],
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            imgAtt_in = tf.layers.dense(inputs=flattenedImgVecs,
                                           units=flattenedImgVecs.get_shape()[-1],
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            #[bx1024]
            
            #duplicate qn vec to combine with each region to get [v_i, q]
            qnAtt_in = tf.expand_dims(qnContext_in, axis=1)
            qnAtt_in = tf.tile(qnAtt_in, [1,tf.shape(flattenedImgVecs)[1],1]) 
            print('Shape of qnAatt_in : {}'.format(qnAtt_in.get_shape()))
            att_in = tf.concat([imgAtt_in, qnAtt_in], axis=-1) #[bx196x1536]
            print('Shape of attention input : {}'.format(att_in.get_shape()))
            
            #compute attention weights
            
            #beta * tanh(wx + b) -- get a scalar val for each region
            print('att_in shape: {}'.format(att_in.get_shape()))
            
            att_f = tf.layers.dense(att_in, units=att_in.get_shape()[-1],
                                activation=tf.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()) #1536
            print('att_f = {}'.format(att_f.get_shape()))
            print('att_f = {}'.format(tf.shape(att_f)))
            beta_w = tf.get_variable("beta", shape=[att_f.get_shape()[-1], 1], dtype=tf.float32) #1536,1
            att_flat = tf.reshape(att_f, shape=[-1, att_f.get_shape()[-1]]) #[b*196, 1536]
            att_flatWeights = tf.matmul(att_flat, beta_w) #get scalar for each batch, region [b*196]
            print('att_flatWeights = {}'.format(att_flatWeights.get_shape()))
            att_regionWeights = tf.reshape(att_flatWeights, shape=[-1, 196])  #[b, 196]
            print('Region weights = {}'.format(att_regionWeights.get_shape()))
            
            #attention function
            if self.config.attentionFunc == 'softmax':
                print('Using softmax attention function')
                self.alpha = tf.nn.softmax(att_regionWeights, name='alpha') # [b,196]
            elif self.config.attentionFunc == 'sigmoid':
                print('Using sigmoid attention function')
                unnorm_alpha = tf.nn.sigmoid(att_regionWeights) #b, 196]
                norm_denominator = tf.expand_dims(
                    tf.reduce_sum(unnorm_alpha, axis=-1), axis=-1) #[b, 1]
                self.alpha = tf.div(unnorm_alpha, norm_denominator, name='alpha') #[b, 196]
            else:
                raise NotImplementedError
            
            alpha = tf.expand_dims(self.alpha, axis=-1)
            
            #compute context: c = sum(alpha) * img
            #broadcast; output shape=[bx1024 or bx1536]
            imgContext = tf.reduce_sum(tf.multiply(alpha, flattenedImgVecs), axis=1) 
            
            return imgContext
        
    
    def construct(self):
        self._addPlaceholders()
        
        self.LSTMinput = self._addEmbeddings()
        
        self.lstmOutput = self._addLSTM(self.LSTMinput) #[batch_size, max_time, 1024]
        
        self.qnContext = self._addQuestionAttention(self.lstmOutput)
            
        self.batch_size = tf.shape(self.img_vecs)[0]
        print('Batch size = {}'.format(self.batch_size))
        
        #reshape image features [bx512x14x14] --> [bx196x512]
        transposedImgVec = tf.transpose(self.img_vecs, perm=[0,3,2,1]) #bx14x14x512
        print('transposedImgVec = {}'.format(transposedImgVec.get_shape()))
        self.flattenedImgVecs = tf.reshape(transposedImgVec, [self.batch_size, 196, 512])
        
        #image attention layer 
        self.imgContext = self._addImageAttention(self.qnContext, self.flattenedImgVecs)
        
        #Combine modes
        if self.config.elMult:
            print('Using pointwise mult')
            #1024 --> 512 or 1536 --> 1024
            attended_img_vecs = tf.layers.dense(inputs=self.imgContext,
                                       units=self.qnContext.get_shape()[-1],
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            #dropout after img mapping layer
            attended_img_vecs = tf.nn.dropout(attended_img_vecs, self.dropout)
                
            self.multimodalOutput = tf.multiply(self.qnContext, attended_img_vecs) #size=512
        else: #using concat
            print('Using concat')
            self.multimodalOutput = tf.concat([self.qnContext, attended_img_vecs], axis=-1)
    
        #fully connected layer
        with tf.variable_scope("proj"):
            hidden_layer2 = tf.layers.dense(inputs=self.multimodalOutput,
                                           units=1000,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.layers.dense(inputs=hidden_layer2,
                                           units=self.config.nOutClasses,
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            print('Shape of y: {}'.format(y.get_shape()))
        #predict & get accuracy
        predProbs = tf.nn.softmax(y)
        self.labels_pred = tf.cast(tf.argmax(predProbs, axis=1), tf.int32, name='labels_pred')
        
        is_correct_prediction = tf.equal(self.labels_pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
        
        self.topK = tf.nn.top_k(predProbs, k=5, name='topK')
                                
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        self.loss = tf.reduce_mean(crossEntropyLoss)

        # Add loss, acc to tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        graph = super(QnAttentionModel, self).loadTrainedModel(restoreModel, restoreModelPath)
        self.alpha = graph.get_tensor_by_name('image_attention/alpha:0')
        if not self.config.noqnatt:
            self.qnAtt_alpha = graph.get_tensor_by_name('qn_attention/qn_alpha:0')
    
    def solve(self, qn, img_id, processor):
        qnAsWordIDsBatch, seqLens, img_vecs = processor.processInput(qn, img_id)
        feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
        }
        qnAlphas, imgAlphas, labels_pred = self.sess.run(
            [self.qnAtt_alpha, self.alpha, self.labels_pred], feed_dict=feed)
        return qnAlphas[0], imgAlphas[0], self.classToAnsMap[labels_pred[0]]
    
    def runPredict(self, valReader, predfile, batch_size=None, mini=False):
        """Evaluates performance on internal valtest set
        Args:
            valReader: 
        Returns:
            metrics:
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if not mini:
            print('Predictions will be logged in {}'.format(predfile))
            self.f2 =  open(predfile, 'wb') 
            self.predFile = csv.writer(self.f2)
            self._logToCSV('Epoch','Question', 'Prediction', 'Label', 'Pred Class',
                 'label class', 'Correct?', 'img id', 'qn_id')
        
        accuracies = []
        correct_predictions, total_predictions = 0., 0.
        img_ids_toreturn, qns_to_return, ans_to_return = [], [], []
        results = []
        for nBatch, (qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids, qn_ids) \
            in enumerate(valReader.getNextBatch(batch_size)):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            if self.config.noqnatt:
                labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            else:
                topK, qnAlphas, alphas, labels_pred = self.sess.run(
                    [self.topK, self.qnAtt_alpha, self.alpha, self.labels_pred], feed_dict=feed)
            
            for lab, labPred, qn, img_id, qn_id in zip(
                labels, labels_pred, rawQns, img_ids, qn_ids):
                if (lab==labPred):
                    correct_predictions += 1
                total_predictions += 1
                accuracies.append(lab==labPred)
                
                currentPred = {}
                currentPred['question_id'] = qn_id
                currentPred['answer'] = self.classToAnsMap[labPred]
                results.append(currentPred)
                
                if not mini:
                    self._logToCSV(nEpoch='', qn=qn, 
                                   pred=self.classToAnsMap[labPred], 
                                   lab=self.classToAnsMap[lab], 
                                   predClass=labPred, labClass=lab, 
                                   correct=lab==labPred, img_id=img_id, qn_id=qn_id)
            
            if mini and nBatch > 1:
                ans_to_return = [self.classToAnsMap[labPred] for labPred in labels_pred]
                img_ids_toreturn = img_ids
                qns_to_return = rawQns
                break
        
        valAcc = np.mean(accuracies)
        print('ValAcc: {:>6.1%}, total_preds: {}'.format(valAcc, total_predictions))
        #return valAcc, correct_predictions, total_predictions
        if mini:
            return qnAlphas, alphas, img_ids_toreturn, qns_to_return, ans_to_return, topK
        return results, valAcc
    