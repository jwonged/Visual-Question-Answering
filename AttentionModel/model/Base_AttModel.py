'''
Created on 20 Mar 2018

@author: jwong
'''

import csv
import json
import os
import pickle
import time
import os

from utils.model_utils import generateForSubmission
import numpy as np
import tensorflow as tf 


class BaseModel(object):
    '''
    Base Model for any of the attention models (ie uses conv5_3 features)
    '''
    def __init__(self, config):
        self.config = config
        
        tf.set_random_seed(self.config.randomSeed)
        self.classToAnsMap = config.classToAnsMap
        self.sess   = None
        self.saver  = None
        self.f1 = None
        self.f2 = None
        print(self._getDescription(config))
    
    def _logToCSV(self, nEpoch='', qn='', pred='', lab='', predClass='', labClass='', 
                  correct='', img_id='', qn_id=''):
        self.predFile.writerow([nEpoch, qn, pred, lab, predClass, labClass,
                                 correct, img_id, qn_id])
    
    def comment(self):
        return ''
    
    def _getDescription(self, config):
        #For logging
        info = 'model: {}, classes: {}, batchSize: {}, \
            dropout: {}, optimizer: {}, lr: {}, decay: {}, \
             clip: {}, shuffle: {}, trainEmbeddings: {}, LSTM_units: {}, \
             usePretrainedEmbeddings: {}, LSTMType: {}, elMult: {}, imgModel: {}, \
             epochsWOimprov: {}, decayAfterEpoch: {}, seed: {}, attentionType: {}, \
             attentionFunc: {}, '.format(
                config.modelStruct, config.nOutClasses, config.batch_size,
                config.dropoutVal, config.modelOptimizer, config.learningRate,
                config.learningRateDecay, config.max_gradient_norm, config.shuffle,
                config.trainEmbeddings, config.LSTM_num_units, config.usePretrainedEmbeddings,
                config.LSTMType, config.elMult, config.imgModel, config.nEpochsWithoutImprov,
                config.decayAfterEpoch, config.randomSeed, config.attentionType, 
                config.attentionFunc)
        return info + 'fc: 2 layers (1000), ' + self.comment() 
    
    def _addOptimizer(self):
        #training optimizer
        with tf.variable_scope("train_step"):
            if self.config.modelOptimizer == 'adam': 
                print('Using adam optimizer')
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.modelOptimizer == 'adagrad':
                print('Using adagrad optimizer')
                optimizer = tf.train.AdagradOptimizer(self.lr)
            else:
                print('Using grad desc optimizer')
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                
            if self.config.max_gradient_norm > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, vs), name='trainModel')
            else:
                self.train_op = optimizer.minimize(self.loss, name='trainModel')
    
    def _initSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(self.config.saveModelPath + 'tensorboard', self.sess.graph)
        
        print('Completed Model Construction')
    
    def train(self, trainReader, valReader, logFile):
        if not self.config.debugMode:
            if not os.path.exists(self.config.saveModelPath):
                os.makedirs(self.config.saveModelPath)
            
        print('Starting model training')
        self.f1 = open(logFile, 'wb')
        self.logFile = csv.writer(self.f1)
        self.logFile.writerow(['Attention model, ', self._getDescription(self.config)])
        self.logFile.writerow([
            'Epoch', 'Val score', 'Train score', 'Train correct', 
            'Train predictions', 'Val correct', 'Val predictions'])
        #self.add_summary()
        highestScore = 0
        nEpochWithoutImprovement = 0
        
        for nEpoch in range(self.config.nTrainEpochs):
            msg = 'Epoch {} \n'.format(nEpoch)
            print(msg)

            score = self._run_epoch(trainReader, valReader, nEpoch)
            if nEpoch > self.config.decayAfterEpoch:
                self.config.learningRate *= self.config.learningRateDecay

            # early stopping and saving best parameters
            if score >= highestScore:
                nEpochWithoutImprovement = 0
                self._saveModel()
                highestScore = score
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochsWithoutImprov:
                    self.logFile.writerow([
                        'Early stopping at epoch {} with {} epochs without improvement'.format(
                            nEpoch+1, nEpochWithoutImprovement)])
                    break
    
    def _run_epoch(self, trainReader, valReader, nEpoch):
        '''
        Runs 1 epoch and returns val score
        '''
        # Potentially add progbar here
        batch_size = self.config.batch_size
        nBatches = trainReader.datasetSize / batch_size
        correct_predictions, total_predictions = 0., 0.
        startTime = time.time()
        
        for i, (qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, _, _) in enumerate(
            trainReader.getNextBatch(batch_size)):
            
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.lr : self.config.learningRate,
                self.dropout : self.config.dropoutVal
            }
            
            if i == 20 and self.config.debugMode:
                _, _, labels_pred, summary, regionWs, exp_regionWs, mask, maskedRWs, denominator, qnalp = self.sess.run(
                [self.train_op, self.loss, self.labels_pred, self.merged,
                 self.qnAtt_regionWeights, self.exp_regionWs,  self.mask, 
                 self.masked_expRegionWs, self.denominator, self.qnAtt_alpha], feed_dict=feed)
                
                print('RegionWs:\n {} \n exp_regionWs:\n {} \n mask:\n {} \n maskedRWs: \n {} \n, denominator: \n {}, \n qnalp: {}\n'.format(
                    regionWs, exp_regionWs, mask, maskedRWs, denominator, qnalp))
                print('RegionWs:{} \n exp_regionWs: {}\n mask: {} \n maskedRWs:{} \n denominator:{}\n qnalp: {}'.format(
                    regionWs.shape, exp_regionWs.shape, mask.shape, maskedRWs.shape, denominator.shape, qnalp.shape))
                
                print('Word IDs: \n{}\n RawQns: {}'.format(qnAsWordIDsBatch, rawQns))
                
            _, _, labels_pred, summary = self.sess.run(
                [self.train_op, self.loss, self.labels_pred, self.merged], feed_dict=feed)
            
            for lab, labPred in zip(labels, labels_pred):
                if lab==labPred:
                    correct_predictions += 1
                total_predictions += 1
                
                #log to csv
                #self.predFile.writerow([qn, self.classToAnsMap[labPred], self.classToAnsMap[lab], labPred, lab, lab==labPred])
                #self.predFile.write('Qn:{}, lab:{}, pred:{}\n'.format(qn, self.classToAnsMap[lab], self.classToAnsMap[labPred]))
                
            if (i%10==0):
                self.tb_writer.add_summary(summary, global_step=nBatches*nEpoch + i)
                                           
            '''valAcc, valCorrect, valTotalPreds = self.runVal(valReader, nEpoch)
            resMsg = 'Epoch {0}, batch {1}: val Score={2:>6.1%}, trainAcc={3:>6.1%}\n'.format(
            nEpoch, i, valAcc, correct_predictions/total_predictions if correct_predictions > 0 else 0 )
            self.logFile.write(resMsg)
            print(resMsg)'''
                
        epochScore, valCorrect, valTotalPreds = self.runVal(valReader, nEpoch)
        trainScore = correct_predictions/total_predictions if correct_predictions > 0 else 0
        
        #logging
        epMsg = 'Epoch {0}: val Score={1:>6.2%}, train Score={2:>6.2%}, total train predictions={3}\n'.format(
                    nEpoch, epochScore, trainScore, total_predictions)
        print(epMsg)
        self.logFile.writerow([nEpoch, epochScore, trainScore, correct_predictions, 
                               total_predictions, valCorrect, valTotalPreds])
        return epochScore
    
    def runVal(self, valReader, nEpoch, is_training=True):
        """Evaluates performance on val set
        Args:
            valReader: 
        Returns:
            metrics:
        """
        accuracies = []
        correct_predictions, total_predictions = 0., 0.
        for qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids,_ in \
            valReader.getNextBatch(self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for lab, labPred, qn ,img_id in zip(labels, labels_pred, rawQns, img_ids):
                if (lab==labPred):
                    correct_predictions += 1
                total_predictions += 1
                accuracies.append(lab==labPred)
                
        valAcc = np.mean(accuracies)
        return valAcc, correct_predictions, total_predictions
    
    def _saveModel(self):
        self.saver.save(self.sess, self.config.saveModelFile)
        
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        print('Restoring model from: {}'.format(restoreModel))
        self.sess = tf.Session()
        self.saver = saver = tf.train.import_meta_graph(restoreModel)
        saver.restore(self.sess, tf.train.latest_checkpoint(restoreModelPath))
        
        graph = tf.get_default_graph()
        self.labels_pred = graph.get_tensor_by_name('labels_pred:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        self.word_ids = graph.get_tensor_by_name('word_ids:0')
        self.img_vecs = graph.get_tensor_by_name('img_vecs:0')
        self.sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        self.dropout = graph.get_tensor_by_name('dropout:0')
        
        if not self.config.notopk:
            self.topK = graph.get_tensor_by_name('topK:0')
            
        self.saver = tf.train.Saver()
        
        return graph 
    
    #overriden in QnAttModel
    def runPredict(self, valReader, predfile, batch_size=None, mini=False):
        """Evaluates performance on internal valtest set
        Args:
            valReader: 
        Returns:
            metrics:
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        print('Predictions will be logged in {}'.format(predfile))
        self.f2 =  open(predfile, 'wb') 
        self.predFile = csv.writer(self.f2)
        self._logToCSV('Epoch','Question', 'Prediction', 'Label', 'Pred Class',
             'label class', 'Correct?', 'img id', 'qn_id')
        
        accuracies = []
        correct_predictions, total_predictions = 0., 0.
        if mini:
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
            alphas, labels_pred = self.sess.run(
                [self.alpha, self.labels_pred], feed_dict=feed)
            
            for lab, labPred, qn, img_id, qn_id in zip(
                labels, labels_pred, rawQns, img_ids, qn_ids):
                if (lab==labPred):
                    correct_predictions += 1
                total_predictions += 1
                accuracies.append(lab==labPred)
                
                self._logToCSV(nEpoch='', qn=qn, 
                               pred=self.classToAnsMap[labPred], 
                               lab=self.classToAnsMap[lab], 
                               predClass=labPred, labClass=lab, 
                               correct=lab==labPred, img_id=img_id, qn_id=qn_id)
                
                currentPred = {}
                currentPred['question_id'] = qn_id
                currentPred['answer'] = self.classToAnsMap[labPred]
                results.append(currentPred)
            
            if mini and nBatch > 1:
                ans_to_return = [self.classToAnsMap[labPred] for labPred in labels_pred]
                img_ids_toreturn = img_ids
                qns_to_return = rawQns
                break
        
        valAcc = np.mean(accuracies)
        print('ValAcc: {:>6.2%}, total_preds: {}'.format(valAcc, total_predictions))
        
        #return valAcc, correct_predictions, total_predictions
        if mini:
            return alphas, img_ids_toreturn, qns_to_return, ans_to_return
        return results, valAcc
        
    
    def runTest(self, testReader, jsonOutputFile):
        '''For producing official test results for submission to server
        '''
        print('Starting test run...')
        allQnIds, allPreds = [], []
        for qnAsWordIDsBatch, seqLens, img_vecs, _, _, qn_ids \
            in testReader.getNextBatch(self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for labPred, qn_id in zip(labels_pred, qn_ids):
                allQnIds.append(qn_id)
                allPreds.append(self.classToAnsMap[labPred])
        
        print('Total predictions: {}'.format(len(allPreds)))
        generateForSubmission(allQnIds, allPreds, jsonOutputFile)
    
    def destruct(self):
        pass
    