'''
Created on 16 Feb 2018

@author: jwong
'''
import numpy as np
import os, sys, getopt
import json
import gc
import pickle

# Main path to caffe installation
caffe_root = '/home/joshua/caffe/'
#caffe_root = '/home/jwong/caffe/'
 
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
#layer_name = 'fc7'
 
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
    #caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(448, 448))
    
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()
    print('Result will be written to {}'.format(outputFile))
    
    #resultJSONData = {}
    errorMessages = []
    
    count = 0
    dataMap = shelve.open(outputFile, flag='n', protocol=pickle.HIGHEST_PROTOCOL)
    print('Extracting from layer: {}'.format(layer_name))
    with open(inputfile, 'r') as reader:
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
                featureData = net.blobs[layer_name].data[0]#.reshape(-1,100352)
                dataMap[str(img_id)] = featureData
                
                #np.savetxt(writer, featureData, fmt='%.8g')
                #resultJSONData[img_id] = featureData
                msg2 = ('\nImages processed: {}\n'.format(count))
            except ValueError:
                print('Error reading image_path')
                errorMessages.append(image_path)
                #Invalid image names
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
    #saveToFile(idlist, idfile)
    #with open(jsonFile, 'w') as jsonOut:
    #    print('writing to {}'.format(jsonFile))
    #    json.dump(resultJSONData, jsonOut)
    #print('Completed {} images'.format(len(resultJSONData)))
    print(errorMessages)

def saveToFile(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to {}'.format(filename))

def checkCorrect():
    #fileName = '../resources/dummyOut.json'
    filename = '../resources/vggTrainConv5_3Features.out'
    print('Reading {}'.format(filename))
    #fp = np.memmap(filename, dtype='float32', mode='r')
    fp = np.loadtxt(filename)
    
    print(fp[0])
    print(fp.shape)
    
    '''with open(fileName) as jsonFile:
        imgData =  json.load(jsonFile)
    
    print(len(imgData))
    #print(imgData[str(359320)][0])
    #print(len(imgData[str(359320)][0]))
    #print(imgData[str(270070)])
    #print(np.asarray(imgData[str(270070)]).shape)
    #print(len(imgData[str(270070)]))'''
    
def main():
    '''
    #train set
    print('Starting processing for training set..')
    inputPath = '../../resources/'
    inputfile = inputPath + 'trainImgPaths.txt'
    #jsonFile = '../resources/vggTrainConv5_3Features.json'
    outputFile = '../resources/vggTrainConv5_3Features_shelf'
    #idFile = '../resources/vggIDsTrainConv5_3Features_save.pkl'
    convertToFeatureVecs(inputPath, inputfile, outputFile)
    print('Training set completed.')
    gc.collect()
    
    #val set
    print('Starting processing for Val set..')
    inputPath = '../../resources/'
    inputfile = inputPath + 'valImgPaths.txt'
    #jsonFile = '../resources/vggValConv5_3Features.json'
    outputFile = '../resources/vggValConv5_3Features_shelf'
    #idFile = '../resources/vggIDsValConv5_3Features.pkl'
    convertToFeatureVecs(inputPath, inputfile, outputFile)
    print('Val set completed.')
    gc.collect()'''
    
    #test set
    print('Starting processing for TEST OFFICIAL set..')
    inputPath = '../../resources/'
    inputfile = inputPath + 'testOfficialImgPaths.txt'
    #jsonFile = '../resources/vggTestOfficialconv5_3Features.json'
    outputFile = '../resources/vggTestOfficialconv5_3Features_shelf'
    #idFile = '../resources/vggIDsTestOfficialconv5_3Features.pkl'
    #jsonFile = '../resources/vggTestOfficialImgFeatures.json'
    convertToFeatureVecs(inputPath, inputfile, outputFile)
    print('Test set completed.')
    
    print('Processing completed!')
    
    '''
    inputPath = '../../resources/'
    inputfile = inputPath + 'testOfficialImgPaths.txt'
    outputfile = '../resources/vggTestOfficialImgFeaturesOut'
    jsonFile = '../resources/vggTestOfficialImgFeatures.json'
    
    convertToFeatureVecs(inputPath, inputfile, outputfile, jsonFile)
    '''

def checkRunOnCPU():
    inputPath = '/media/jwong/Transcend/VQADataset/TrainSet/'
    inputfile = inputPath + 'trainImgPaths.txt'
    
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(448, 448))
    
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
            if count == 1:
                break
            img_id, featureData, msg = convertImageToVec(image_path, labels, net)
            if img_id == -1:
                errorMessages.append(msg)
                continue
            resultJSONData[img_id] = featureData
            #print(featureData)
            print(np.asarray(featureData).shape)
            msg2 = ('\nImages processed: {}\n'.format(count))
            print(msg2)
            print(layer_name)
            count = count + 1
            
            
def convertImageToVec(image_path, labels, net):
    image_path = image_path.strip()
    input_image = caffe.io.load_image(image_path)
    print(input_image)
    print(input_image.shape)
    prediction = net.predict([input_image], oversample=False)
    msg = ('{} : {} ( {} )'.format(os.path.basename(image_path), 
                                   labels[prediction[0].argmax()].strip(), 
                                   prediction[0][prediction[0].argmax()]))
    
    try:
        img_id = getImageID(image_path)
        featureData = net.blobs[layer_name].data[0].tolist()#.reshape(1,-1).tolist()
        return img_id, featureData, msg

    except ValueError:
        print('Error reading image_path'.format(image_path))
        return -1, None, image_path

if __name__ == '__main__':
    if (sys.argv[1] == '-GPU'):
        main()
    elif (sys.argv[1] == '-CPU'):
        checkRunOnCPU()
    elif (sys.argv[1] == '-check'):
        checkCorrect()
    else:
        print('Use option -GPU or -CPU')
    