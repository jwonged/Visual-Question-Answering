'''
Created on 16 Feb 2018

@author: jwong
'''
import numpy as np
import os, sys, getopt
import json

# Main path to caffe installation
caffe_root = '/home/joshua/caffe/'#'/home/jwong/caffe/'
 
# Model prototxt file
model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
 
# Model caffemodel file
model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
 
# Name of the layer we want to extract
layer_name = 'fc7'
 
sys.path.insert(0, caffe_root + 'python')

import caffe

def getImageID(image_path):
    #Get image ID
    splitPath = image_path.split('/')
    imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
    suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
    img_id = int(suffix.split('.')[0])
    print 'Reading img ', img_id
    return img_id
    

def main():
    inputPath = '../../resources/'
    inputfile = inputPath + 'trainImgPaths.txt'
    outputfile = '../resources/vggTrainImgFeaturesOut'
    jsonFile = '../resources/vggTrainImgFeatures.json'
    
    count = 0
    with open(inputfile, 'r') as reader:
        for path in reader:
            count += 1
    print('Preparing to read {} images'.format(count))
    
    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(480, 480))
    
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    
    resultJSONData = {}
    errorMessages = []
    
    count = 0
    with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            for image_path in reader:
                image_path = image_path.strip()
                input_image = caffe.io.load_image(inputPath + image_path)
                prediction = net.predict([input_image], oversample=False)
                print (os.path.basename(image_path), ' : ' , \
                       labels[prediction[0].argmax()].strip() , \
                       ' (', prediction[0][prediction[0].argmax()] , ')')
                count = count + 1
                try:
                    img_id = getImageID(image_path)
                    
                    # filename, array data to be saved, format, delimiter
                    featureData = net.blobs[layer_name].data[0].reshape(1,-1).tolist()
                    np.savetxt(writer, featureData, fmt='%.8g')
                    resultJSONData[img_id] = featureData
                    print 'Images processed: {}'.format(count)
    
                except ValueError:
                    print('Error reading image_path')
                    errorMessages.append(image_path)
                    #Invalid image names
                    errorMessages.append(image_path)
    
    with open(jsonFile, 'w') as jsonOut:
        print('writing to {}'.format(jsonFile))
        json.dump(resultJSONData, jsonOut)
    print('Completed {} images'.format(len(resultJSONData)))
    print(errorMessages)
                    
def checkCorrect():
    fileName = '../resources/dummyOut.json'
    print('Reading {}'.format(fileName))
    with open(fileName) as jsonFile:
        imgData =  json.load(jsonFile)
    
    print(len(imgData))
    print(imgData[str(359320)][0])
    print(len(imgData[str(359320)][0]))
    
    
if __name__ == '__main__':
    main()