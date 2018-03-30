'''
Created on 30 Mar 2018

@author: jwong
'''

import caffe
import sys, os
import numpy as np
class ImageProcessor(object):
    '''
    Processing images for E2E architecture
    '''

    def __init__(self):
        caffe_root = '/home/joshua/caffe/'
        #caffe_root = '/home/jwong/caffe/'
         
        # Model prototxt file
        self.model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
         
        # Model caffemodel file
        self.model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
         
        # File containing the class labels
        self.imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
         
        # Path to the mean image (used for input processing)
        self.mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
         
        # Name of the layer we want to extract
        self.layer_name = 'conv5_3'
        #layer_name = 'fc7'
         
        sys.path.insert(0, caffe_root + 'python')
        
        caffe.set_device(0)
        #caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        self.net = caffe.Classifier(self.model_prototxt, self.model_trained,
                           mean=np.load(self.mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(448, 448))
        with open(self.imagenet_labels) as f:
            self.labels = f.readlines()
    
    def _getImageID(self, image_path):
        #Get image ID
        splitPath = image_path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def processSingleImage(self, imageFile, getID=False):
        print('Extracting from layer: {}'.format(self.layer_name))
        input_image = caffe.io.load_image(imageFile.strip())
        prediction = self.net.predict([input_image], oversample=False)
        msg = ('{} : {} ( {} )'.format(imageFile.split('/')[-1], 
                                       self.labels[prediction[0].argmax()].strip(), 
                                       prediction[0][prediction[0].argmax()]))
        featureData = self.net.blobs[self.layer_name].data[0]
        
        if getID:
            img_id = self._getImageID(imageFile)
        else:
            img_id = -1
            
        return featureData, img_id