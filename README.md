# Visual Question Answering
AttentionModel: code for my Image and Question Attention models, and crossmodal attention    
BOWIMG: the bag-of-words + pre-trained GoogLeNet/VGGNet convolutional NN model   
DatasetPreprocessor: code for preprocessing the VQA dataset and other quick scripts    
LSTMCNN: Bi-LSTM + CNN (from scratch) model    
LSTMIMG: Bi-LSTM + pre-trained GoogLeNet model      
Preparation: code written during the preparation phase e.g. practising tensorflow    
WebApp: Web application which runs the image attention model    

# Dataset
http://visualqa.org/

# Requirements
Tensorflow - https://www.tensorflow.org/install/install_linux   
NLTK - http://www.nltk.org/install.html   
opencv (for shallow CNN) - https://docs.opencv.org/trunk/d2/de6/tutorial_py_setup_in_ubuntu.html   
caffe (for pre-trained CNN) - http://caffe.berkeleyvision.org/install_apt.html   

## Resources

To get a file 'images.txt' containing the paths of all images in the folder /images:   
find `pwd`/images -type f -exec echo {} \; > images.txt   
   

### Word embeddings
Word2Vec model - https://code.google.com/archive/p/word2vec/  
gensim - https://radimrehurek.com/gensim/models/word2vec.html  
GloVe - https://nlp.stanford.edu/projects/glove/   
Others - http://ahogrammer.com/2017/01/20/the-list-of-pretrained-word-embeddings/   

### CNN Models
GoogLeNet - http://www.marekrei.com/blog/transforming-images-to-feature-vectors/    
VGGnet - https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md   
Other models - http://caffe.berkeleyvision.org/model_zoo.html   


### For installing caffe
https://shreyasskandan.github.io/caffe_linux_blogpost.html
https://chunml.github.io/ChunML.github.io/project/Installing-Caffe-CPU-Only/
https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide   
   
For GPU, follow Makefile Config in:   
https://gist.github.com/nikitametha/c54e1abecff7ab53896270509da80215   

Common errors:   
/usr/bin/ld: cannot find -lopencv_imgcodecs   
/usr/bin/ld: cannot find -lopencv_imgcodecs   
collect2: error: ld returned 1 exit status   
Makefile:573: recipe for target '.build_release/lib/libcaffe.so.1.0.0' failed   
make: *** [.build_release/lib/libcaffe.so.1.0.0] Error 1   

One of these might fix it: either add -lopencv_imgcodecs in Makefile (not Makefile.config)
or uncomment USE\_OPENCV:=0 or correct OPENCV\_VERSION:=3 or uncomment USE\_PKG\_CONFIG:=1   

Also comment out -gencode arch=compute\_20 (depending on CUDA version)   
   
PYTHON_INCLUDE := /usr/include/python2.7 \
        /usr/local/lib/python2.7/dist-packages/numpy/core/include

