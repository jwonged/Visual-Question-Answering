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
    BOW+IMG base_model
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
        info = 'model: {}, classes: {}, batchSize: {}, dropout: {}, \
            optimizer: {}, lr: {}, decay: {}, shuffle: {}, trainEmbeddings: {}, \
             usePretrainedEmbeddings: {},   imgModel: {}, \
             epochsWOimprov: {}, decayAfterEpoch: {}, seed: {},'.format(
                 config.modelStruct, config.nOutClasses, config.batch_size,
                 config.dropoutVal, config.modelOptimizer, config.learningRate,
                 config.learningRateDecay, config.shuffle, config.trainEmbeddings,  
                 config.usePretrainedEmbeddings, config.imgModel, 
                 config.nEpochsWithoutImprov, config.decayAfterEpoch, config.randomSeed)
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
                
            self.train_op = optimizer.minimize(self.loss, name='trainModel')
    
    def _initSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(self.config.saveModelPath + 'tensorboard', self.sess.graph)
        
        print('Completed Model Construction')
    
    def train(self, trainReader, valReader, logFile):
        if not os.path.exists(self.config.saveModelPath):
            os.makedirs(self.config.saveModelPath)
            
        print('Starting model training')
        self.f1 = open(logFile, 'wb')
        self.logFile = csv.writer(self.f1)
        self.logFile.writerow(['Attention model, ', self._getDescription(self.config)])
        self.logFile.writerow([
            'Epoch', 'Val score', 'Train score', 'Train correct', 
            'Train predictions', 'Val correct', 'Val predictions'])
        startTime = time.time()
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
        
        timeTaken = time.time() - startTime
        print('Time taken: {}'.format(timeTaken))
        self.logFile.writerow(['Time taken', str(timeTaken)])
        
    
    def _run_epoch(self, trainReader, valReader, nEpoch):
        '''
        Runs 1 epoch and returns val score
        '''
        # Potentially add progbar here
        batch_size = self.config.batch_size
        nBatches = trainReader.datasetSize / batch_size
        correct_predictions, total_predictions = 0., 0.
        startTime = time.time()
        
        for i, (qnAsWordIDsBatch, seqLens, img_vecs, labels, _, _, _) in enumerate(
            trainReader.getNextBatch(batch_size)):
            
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.lr : self.config.learningRate,
                self.dropout : self.config.dropoutVal
            }
            
            if i==0 and self.config.debugMode:
                we, fm, me, bow, mm, _, _, labels_pred, summary = self.sess.run(
                [self.word_embeddings, self.float_mask,
                  self.mask_embeddings, self.bows, self.multimodalOutput,
                  self.train_op, self.loss, self.labels_pred, self.merged], feed_dict=feed)
                
                print('word_em: \n {} \n floatmask: \n {} \n maskem: \n {} \n bow: \n {} \n'.format(
                    we, fm, me, bow))
                print('word_em: {} \n floatmask: {} \n maskem: {} \n bow: {} \n mm: {}\n'.format(
                    we.shape, fm.shape, me.shape, bow.shape, mm.shape))
            
            _, _, labels_pred, summary = self.sess.run(
                [self.train_op, self.loss, self.labels_pred, self.merged], feed_dict=feed)
            
            
            for lab, labPred in zip(labels, labels_pred):
                if lab==labPred:
                    correct_predictions += 1
                total_predictions += 1
                
            if (i%10==0):
                self.tb_writer.add_summary(summary, global_step=nBatches*nEpoch + i)
                          
                
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
        tf.reset_default_graph()
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
        self.topK = graph.get_tensor_by_name('topK:0')
        
        self.saver = tf.train.Saver()
        
        return graph 
        
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
        results = []
        correct_predictions, total_predictions = 0., 0.
        for nBatch, (qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids, qn_ids) \
            in enumerate(valReader.getNextBatch(batch_size)):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
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
            
        valAcc = np.mean(accuracies)
        print('ValAcc: {:>6.2%}, total_preds: {}'.format(valAcc, total_predictions))
        return results, valAcc
    
    def runEvaluationMetrics(self, valReader):
        batch_size = self.config.batch_size
        
        allPredictions = []
        allLabels = []
        for nBatch, (qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids, qn_ids) \
            in enumerate(valReader.getNextBatch(batch_size)):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            allPredictions += labels_pred.tolist()
            allLabels += labels
        
        print('Completed {} predictions'.format(len(allPredictions)))
        print('{} labels'.format(len(allLabels)))
        
        return allLabels, allPredictions, self.classToAnsMap
    
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
        self.sess.close()
    
    