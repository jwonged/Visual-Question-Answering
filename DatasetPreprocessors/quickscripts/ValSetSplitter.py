import numpy as np
import os, sys, getopt
import json
 
# Main path to caffe installation
caffe_root = '/home/jwong/caffe/'
 
# Model prototxt file
model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
 
# Model caffemodel file
model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
 
# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
 
# Name of the layer we want to extract
layer_name = 'pool5/7x7_s1'
 
sys.path.insert(0, caffe_root + 'python')
import caffe
 
def main():
    inputfile = '/media/jwong/Transcend/VQADataset/ValSet/valImagePaths.txt'
    outputfile = '/media/jwong/Transcend/VQADataset/ValSet/VQAValOutput2.out'
    jsonValFile = '/media/jwong/Transcend/VQADataset/ValSet/VQAImgFeatures_Val.json'
    errorLogFile = '/media/jwong/Transcend/VQADataset/ValSet/ImgFeatures_TrainLog.txt'

    print 'Reading images from "', inputfile
 
    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(480, 640))
 
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()

    resultJSONData = {}
    errorMessages = []
    
    count = 0
    # Processing one image at a time, print predictions
    with open(inputfile, 'r') as reader:
        with open(outputfile, 'w') as writer:
            writer.truncate()
            for image_path in reader:
                if (count <= 20252):
                    count = count + 1
                    if (count%1000 == 0):
                        print('Passing count:' + str(count))
                    continue

                image_path = image_path.strip()
                input_image = caffe.io.load_image(image_path)
                prediction = net.predict([input_image], oversample=False)
                print os.path.basename(image_path), ' : ' , labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
                count = count + 1
                try:
                    #Get image ID
                    splitPath = image_path.split('/')
                    imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
                    suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
                    img_id = int(suffix.split('.')[0])
                    print 'Reading img ', img_id
                    print 'Images processed : ', count

                    # filename, array data to be saved, format, delimiter
                    featureData = net.blobs[layer_name].data[0].reshape(1,-1).tolist()
                    np.savetxt(writer, featureData, fmt='%.8g')
                    resultJSONData[img_id] = featureData

                except ValueError:
                    #Invalid image names
                    errorMessages.append(image_path)

    with open(jsonValFile, 'w') as jsonOut:
        json.dump(resultJSONData, jsonOut)
    with open(errorLogFile, 'w') as logFile:
        for msg in errorMessages:
            logFile.write(msg)
 
if __name__ == "__main__":
    main()

#start from 000000062154.jpg
#000000034372.jpg