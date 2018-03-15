'''
Created on 15 Mar 2018

@author: jwong
'''

import numpy as np
import os, sys, getopt
import json
import gc
import pickle

# Main path to caffe installation
caffe_root = '/home/joshua/caffe/'
 
# Model prototxt file
model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
 
# Model caffemodel file
model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
 
# Name of the layer we want to extract
layer_name = 'conv5_3'
 
sys.path.insert(0, caffe_root + 'python')

import caffe
import shelve

def getImageID(image_path):
    #Get image ID
    splitPath = image_path.split('/')
    imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
    suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
    img_id = int(suffix.split('.')[0])
    return img_id
    
def convertToFeatureVecs(inputPath, inputfile, outputFile):
    count = 0
    with open(inputfile, 'r') as reader:
        for path in reader:
            count += 1
    print('Preparing to read {} images'.format(count))
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(448, 448))
    
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    
    print('Results edited in: {}'.format(outputFile))
    
    errorMessages = []
    
    countDone = 0
    count
    dataMap = shelve.open(outputFile, flag='w', protocol=pickle.HIGHEST_PROTOCOL)
    print(len(dataMap))
    dataMap.close()
    exit()
    print('Extracting from layer: {}'.format(layer_name))
    with open(inputfile, 'r') as reader:
        for image_path in reader:
            image_path = image_path.strip()
            
            img_id = getImageID(image_path)
            if (str(img_id) in dataMap):
                if countDone%10==0:
                    print('Contains {}, count {}'.format(img_id, countDone))
                countDone += 1
                continue
            
            input_image = caffe.io.load_image(inputPath + image_path)
            prediction = net.predict([input_image], oversample=False)
            msg = ('{} : {} ( {} )'.format(os.path.basename(image_path), 
                                           labels[prediction[0].argmax()].strip(), 
                                           prediction[0][prediction[0].argmax()]))
            
            count = count + 1
            
            try:
                
                # filename, array data to be saved, format, delimiter
                featureData = net.blobs[layer_name].data[0]
                dataMap[str(img_id)] = featureData
                
                msg2 = ('\nImages processed: {}\n'.format(count))
            except ValueError:
                print('Error reading image_path')
                errorMessages.append(image_path)
            
            if count%200 == 0:
                print(featureData.shape)
                print(msg)
                print(msg2)
            if count%1000 == 0:
                print('Doing a data sync...')
                dataMap.sync()
                print('Data sync done.')
    dataMap.close()
                    
    print('Completed processing {} images'.format(count))
    print('Error messages: {}'.format(errorMessages))


def main():
    print('Starting processing for TEST OFFICIAL set..')
    inputPath = '../../resources/'
    inputfile = inputPath + 'testOfficialImgPaths.txt'
    outputFile = '../resources/vggTestOfficialconv5_3Features_shelf'
    convertToFeatureVecs(inputPath, inputfile, outputFile)
    print('Test set completed.')
    print('Processing completed!')
    

if __name__ == '__main__':
    main()
    