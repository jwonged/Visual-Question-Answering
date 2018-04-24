'''
@author: jwong
'''

import matplotlib.pyplot as plt
import skimage.transform
import cv2
from scipy import ndimage
import numpy as np
from nltk import word_tokenize
from textwrap import wrap
import pickle

class OutputGenerator(object):
    def __init__(self, imgPathsFile):
        self.idToImgpathMap = {}
        print('Reading {}'.format(imgPathsFile))
        with open(imgPathsFile, 'r') as reader:
            for image_path in reader:
                image_path = image_path.strip()
                self.idToImgpathMap[str(self.getImageID(image_path))] = image_path
        self.alphaMap = None
    
    def getImageID(self,image_path):
        #Get image ID
        splitPath = image_path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def convertIDtoPath(self, img_id):
        return self.idToImgpathMap[img_id]
    
    def displayQnImgAttention(self, qnAlphas, imgAlphas, img_ids, qns, preds, topk, labs, saveData=False):
        if saveData:
            data = {}
            qn_alphas_to_save = []
            alphas_to_save, images_to_save, preds_to_save, qns_to_save, labs_to_save = [], [], [], [], []
        for n, (qnAl, imAl, img_id, qn, pred, topk, lab) in enumerate(zip(
            qnAlphas, imgAlphas, img_ids, qns, preds, topk, labs)):
            #if n > 6:
            #    break
            
            alp_img = self._processImgAlpha(imAl)
            toks = word_tokenize(qn)
            for tok, att in zip(toks, qnAl):
                print('{} ( {} )  '.format(tok,att))
            #imgvec = cv2.imread(self.idToImgpathMap[img_id])
            #imgvec = cv2.resize(imgvec, dsize=(448,448))
            #from PIL import Image
            #img = Image.open(self.idToImgpathMap[img_id])
            #img.show()
            imgvec = self._readImageAndResize(self.idToImgpathMap[img_id])
            qn_2d = np.expand_dims(qnAl[:len(toks)], axis=0)
            
            if saveData:
                alphas_to_save.append(imAl)
                qn_alphas_to_save.append(qn_2d)
                images_to_save.append(imgvec)
                preds_to_save.append(pred)
                qns_to_save.append(qn)
                labs_to_save.append(lab)
            
            
            print(qn_2d)
            plt.suptitle('Qn: {}\n Prediction: {} \n Ground truth: {}'.format(qn, pred, lab))#, fontsize=16)
            
            #attended image
            plt.subplot(2,2,1)
            #plt.title('')
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80)#, cmap='gray')
            plt.axis('off')
            
            #img
            plt.subplot(2,2,2)
            #plt.title('Qn: {}, pred: {}'.format(qn, pred))
            plt.imshow(imgvec)
            plt.axis('off')
            
            #qn attention
            plt.subplot(2,1,2)
            plt.yticks([])
            plt.xticks(np.arange(len(toks)), (toks))
            plt.imshow(qn_2d, cmap='gray_r', interpolation='nearest')
            plt.show()
        
        if saveData:
            data['qn_alphas'] = qn_alphas_to_save
            data['im_alphas'] = alphas_to_save
            data['images'] = images_to_save
            data['preds'] = preds_to_save
            data['qns'] = qns_to_save
            data['labs'] = labs_to_save
            self._saveToPickle(data, fileName='/media/jwong/Transcend/VQAresults/Samplespictures/QuAttsigmoidValFirst.pkl')
            
    def _saveToPickle(self, data, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved to {}'.format(fileName))
    
    def _getAlphaMap(self):
        alphaMapFile = './utils/alphaMap.pkl'
        print('Reading ' +  alphaMapFile)
        with open(alphaMapFile, 'rb') as f:
            data = pickle.load(f)
        return data
        
    def _processImgAlpha(self, imgAlpha):
        
        alp_img = skimage.transform.pyramid_expand(
            imgAlpha.reshape(14,14), upscale=16, sigma=20)
        #alp_img = skimage.transform.resize(
        #    imgAlpha.reshape(14,14), [448, 448])
        alp_img = np.transpose(alp_img, (1,0))
        
        return alp_img
    
    def _processImgAlphaDeconv(self, imgAlpha):
        imgAlpha = np.asarray(imgAlpha.reshape(14,14))
        alp_img = np.zeros((224,224))
        for i in range(224):
            for j in range(224):
                count = 0
                for tuplet in self.alphaMap[i][j]:
                    yo = imgAlpha[tuplet[0]][tuplet[1]]
                    alp_img[i,j] += yo
                    count += 1
                alp_img[i, j] = alp_img[i, j] / count
        alp_img = np.transpose(alp_img, (1,0))
        return alp_img
    
    def _readImageAndResize(self, path):
        '''
        from skimage.transform import resize
        import skimage.io
        img = skimage.img_as_float(skimage.io.imread(
            path, as_grey=False)).astype(np.float32)
        im_min, im_max = img.min(), img.max()
        im_std = (img - im_min) / (im_max - im_min)
        resized_std = resize(im_std, (448,448), order=1, mode='constant')
        resized_im = resized_std * (im_max - im_min) + im_min
        imgvec = resized_im
        '''
        from PIL import Image
        img = Image.open(path)
        img = img.resize((224,224))
        imgvec = np.asarray(img)
        #plt.imread(path)
        return imgvec
    
    def displaySingleOutput(self, alpha, img_id, qn, pred):
        print('Num of images: {}'.format(img_id))
        
        imgvec = self._readImageAndResize(self.idToImgpathMap[img_id])
        
        #imgvec = cv2.imread(self.idToImgpathMap[img_id])
        #imgvec = ndimage.imread(self.idToImgpathMap[img_id])
        #imgvec = cv2.resize(imgvec, dsize=(448,448))
        
        alp_img = self._processImgAlpha(alpha)
        
        plt.subplot(1,1,1)
        plt.title('Qn: {}, pred: {}'.format(qn, pred))
        plt.imshow(imgvec)
        plt.imshow(alp_img, alpha=0.80)
        plt.axis('off')
        
        #plt.subplot(2,1,1)
        #plt.title('Qn: {}, pred: {}'.format(qn, pred))
        #plt.imshow(imgvec)
        #plt.axis('off')
            
        plt.show()
        
    def displayOutput(self, alphas, img_ids, qns, preds):
        print('Num of images: {}'.format(img_ids))
        fig = plt.figure()
        for n, (alp, img_id, qn, pred) in enumerate(zip(alphas, img_ids, qns, preds)):
            if n>2:
                break
            
            imgvec = self._readImageAndResize(self.idToImgpathMap[img_id])
            #imgvec = cv2.imread(self.idToImgpathMap[img_id])
            #imgvec = ndimage.imread(self.idToImgpathMap[img_id])
            #imgvec = cv2.resize(imgvec, dsize=(448,448))
            
            alp_img = self._processImgAlpha(alp)
            
            plt.subplot(2,3,(n+1))
            plt.title("\n".join(wrap(
                "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80) #plt.imshow(arr, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2,3,(n+1)*2)
            plt.title("\n".join(wrap(
                "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            plt.axis('off')
            
            
            '''
            plt.subplot(2,4,(n+1))
            plt.title("\n".join(wrap(
                "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80) #plt.imshow(arr, cmap='gray')
            plt.axis('off')
            
            plt.subplot(2,4,(n+1)*2)
            plt.title('Qn: {}, pred: {}'.format(qn, pred))
            plt.imshow(imgvec)
            plt.axis('off')'''
        #plt.tight_layout()
        plt.show()
        
    def displayEachSample(self, alphas, img_ids, qns, preds, labs, saveData=False):
        print('Num of images: {}'.format(img_ids))
        if saveData:
            data = {}
            alphas_to_save, images_to_save, preds_to_save, qns_to_save, labs_to_save = [], [], [], [], []
        for n, (alp, img_id, qn, pred, lab) in enumerate(zip(alphas, img_ids, qns, preds, labs)):
            imgvec = self._readImageAndResize(self.idToImgpathMap[img_id])
            #imgvec = cv2.imread(self.idToImgpathMap[img_id])
            #imgvec = ndimage.imread(self.idToImgpathMap[img_id])
            #imgvec = cv2.resize(imgvec, dsize=(448,448))
            
            alp_img = self._processImgAlpha(alp)
            
            if saveData:
                alphas_to_save.append(alp_img)
                images_to_save.append(imgvec)
                preds_to_save.append(pred)
                qns_to_save.append(qn)
                labs_to_save.append(lab)
            
            plt.subplot(1,2,1)
            #plt.xlabel('Qn: {} \nPred: {}'.format(qn, pred))
            #plt.title("\n".join(wrap(
            #    "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            
            plt.axis('off')
            plt.suptitle('Qn: {}\n Pred: {} \n Ans: {}'.format(qn, pred, lab), fontsize=16)
            
            plt.subplot(1,2,2)
            #plt.title("\n".join(wrap(
            #    "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80) #plt.imshow(arr, cmap='gray')
            plt.axis('off')
            plt.show()
        #plt.tight_layout()
        
        if saveData:
            data['alphas'] = alphas_to_save
            data['images'] = images_to_save
            data['preds'] = preds_to_save
            data['qns'] = qns_to_save
            data['labs'] = labs_to_save
            self._saveToPickle(data, fileName='/media/jwong/Transcend/VQAresults/Samplespictures/ImgAttSamplesSig40.pkl')
            
            
        