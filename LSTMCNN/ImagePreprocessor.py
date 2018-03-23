'''
Created on 7 Feb 2018

@author: jwong
'''

import pickle
import cv2 as cv
import numpy as np

class ImagePreprocessor(object):
    '''
    Input: txt file of img paths
    Output: pkl map of img_id --> raw img pixel vec
    '''

    def __init__(self):
        pass
    
    def _getImageID(self, path):
        path = path.strip()
        splitPath = path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def _getImgPixelVec(self, path):
        img = cv.imread(path, 1)
        img = cv.resize(img, (128, 128),0,0, cv.INTER_AREA)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)
        return img
    
    def processImages(self, inputfile):
        imgMap = {}
        with open(inputfile, 'r') as reader:
            for image_path in reader:
                image_path = image_path.strip()
                imgMap[self._getImageID(image_path)] = self._getImgPixelVec(image_path)
        
        print('Converted {} images from {}'.format(len(imgMap), inputfile))
        return imgMap

    def saveToFile(self, data, filepkl):
        with open(filepkl, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved to {}'.format(filepkl))
        
if __name__ == "__main__":
    trainfile = '/media/jwong/Transcend/VQADataset/TrainSet/trainImgPaths.txt'
    valfile = '/media/jwong/Transcend/VQADataset/ValTestSet/valImgPaths.txt'
    
    processor = ImagePreprocessor()
    trainpkl = '/media/jwong/Transcend/VQADataset/CNNImgMaps/trainImgPixelMap128.pkl'
    trainMap = processor.processImages(trainfile)
    processor.saveToFile(trainMap, trainpkl)
    
    valpkl = '/media/jwong/Transcend/VQADataset/CNNImgMaps/trainImgPixelMap128.pkl'
    valMap = processor.processImages(valfile)
    processor.saveToFile(valMap, valpkl)
    
    
    
    