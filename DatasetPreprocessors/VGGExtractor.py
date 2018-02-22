'''
Created on 16 Feb 2018

@author: jwong
'''
import numpy as np
import os, sys, getopt
import json

# Main path to caffe installation
#caffe_root = '/home/joshua/caffe/'
caffe_root = '/home/jwong/caffe/'
 
# Model prototxt file
model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
 
# Model caffemodel file
model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
 
# Name of the layer we want to extract
layer_name = 'pool5' #'fc7'
 
sys.path.insert(0, caffe_root + 'python')

import caffe

def getImageID(image_path):
    #Get image ID
    splitPath = image_path.split('/')
    imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
    suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
    img_id = int(suffix.split('.')[0])
    return img_id
    

def convertToFeatureVecs(inputPath, inputfile, outputfile, jsonFile):
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
    #exit()
    
    resultJSONData = {}
    errorMessages = []
    
    count = 0
    with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            writer.truncate()
            for image_path in reader:
                image_path = image_path.strip()
                input_image = caffe.io.load_image(inputPath + image_path)
                prediction = net.predict([input_image], oversample=False)
                msg = ('{} : {} ( {} )'.format(os.path.basename(image_path), 
                                               labels[prediction[0].argmax()].strip(), 
                                               prediction[0][prediction[0].argmax()]))
                
                count = count + 1
                
                try:
                    img_id = getImageID(image_path)
                    
                    # filename, array data to be saved, format, delimiter
                    featureData = net.blobs[layer_name].data[0].reshape(1,-1).tolist()
                    #np.savetxt(writer, featureData, fmt='%.8g')
                    print(featureData)
                    print(np.asarray(featureData).shape)
                    resultJSONData[img_id] = featureData
                    msg2 = ('\nImages processed: {}\n'.format(count))
    
                except ValueError:
                    print('Error reading image_path')
                    errorMessages.append(image_path)
                    #Invalid image names
                    errorMessages.append(image_path)
                
                if count%200 == 0:
                    print(msg)
                    print(msg2)
    
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
    
def main():
    '''
    #val set
    inputPath = '../../resources/'
    inputfile = inputPath + 'valImgPaths.txt'
    outputfile = '../resources/vggValImgFeaturesOut'
    jsonFile = '../resources/vggValImgFeatures.json'
    
    convertToFeatureVecs(inputPath, inputfile, outputfile, jsonFile)
    '''
    inputPath = '../../resources/'
    inputfile = inputPath + 'testOfficialImgPaths.txt'
    outputfile = '../resources/vggTestOfficialImgFeaturesOut'
    jsonFile = '../resources/vggTestOfficialImgFeatures.json'
    
    convertToFeatureVecs(inputPath, inputfile, outputfile, jsonFile)

def checkRunOnCPU():
    inputPath = '/media/jwong/Transcend/VQADataset/TrainSet/'
    inputfile = inputPath + 'trainImgPaths.txt'
    
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(480, 480))
    
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    
    print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()
    
    resultJSONData = {}
    errorMessages = []
    
    count = 0
    with open(inputfile, 'r') as reader:
        for image_path in reader:
            if count == 3:
                break
            image_path = image_path.strip()
            input_image = caffe.io.load_image(image_path)
            prediction = net.predict([input_image], oversample=False)
            msg = ('{} : {} ( {} )'.format(os.path.basename(image_path), 
                                           labels[prediction[0].argmax()].strip(), 
                                           prediction[0][prediction[0].argmax()]))
            
            count = count + 1
            
            try:
                img_id = getImageID(image_path)
                featureData = net.blobs[layer_name].data[0].reshape(1,-1).tolist()
                print(featureData)
                print(np.asarray(featureData).shape)
                resultJSONData[img_id] = featureData
                msg2 = ('\nImages processed: {}\n'.format(count))
                print(msg2)

            except ValueError:
                print('Error reading image_path')
                errorMessages.append(image_path)
                #Invalid image names
                errorMessages.append(image_path)
            
    

    
if __name__ == '__main__':
    checkRun()