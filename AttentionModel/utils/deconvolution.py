'''
Created on 1 Apr 2018

@author: jwong
'''

import pickle

def mkIm(dim=4):
    im = {}
    for i in range(dim):
        im[i] = {}
        for j in range(dim):
            im[i][j] = [(i,j)]
    return im

def pooling(prev):
    #2x2 window, stride=2
    rows = len(prev)
    cols = len(prev[0])
    
    result = {}
    for i in range(0, rows, 2):
        result[i/2] = {}
        for j in range(0, cols, 2):
            newElement = set()
            
            newElement = newElement.union(prev[i][j])
            newElement = newElement.union(prev[i+1][j])
            newElement = newElement.union(prev[i][j+1])
            newElement = newElement.union(prev[i+1][j+1])
            
            result[i/2][j/2] = newElement
    return result

def convdic(prev):
    #3x3 kernel, stride=1 pixel, padding=1
    rows = len(prev)
    cols = len(prev[0])
    
    result = {}
    for i in range(rows):
        result[i] = {}
        for j in range(cols):
            newElement = set()
            
            #top 3
            if i > 0 and j > 0:
                newElement = newElement.union(prev[i-1][j-1])
            if i > 0:
                newElement = newElement.union(prev[i-1][j])
            if i > 0 and j < cols-1:
                newElement = newElement.union(prev[i-1][j+1])
                
            #center
            if j > 0:
                newElement = newElement.union(prev[i][j-1])
            newElement = newElement.union(prev[i][j])
            if j < rows-1:
                newElement = newElement.union(prev[i][j+1])
                
            #bottom
            if j > 0 and i < rows-1:
                newElement = newElement.union(prev[i+1][j-1])
            if i < rows-1:
                newElement = newElement.union(prev[i+1][j])
            if i < rows-1 and j < cols-1:
                newElement = newElement.union(prev[i+1][j+1])
            
            result[i][j] = newElement
    return result

def reVGGNet19():
    """
    returns a 14x14 key dictionary
    element[i][j] is a set of all initial pixel coordinates that it was created from
    """
    im = mkIm(224)
    conv1 = convdic(im)
    conv2 = convdic(conv1)
    pool1 = pooling(conv2)
    
    conv3 = convdic(pool1)
    conv4 = convdic(conv3)
    pool2 = pooling(conv4)
    
    conv5 = convdic(pool2)
    conv6 = convdic(conv5)
    conv7 = convdic(conv6)
    conv8 = convdic(conv7)
    pool3 = pooling(conv8)
    print('Pool 3: {}'.format(len(pool3)))
    print('Pool 3 set: {}'.format(len(pool3[5][5])))
    
    conv9 = convdic(pool3)
    conv10 = convdic(conv9)
    conv11 = convdic(conv10)
    conv12 = convdic(conv11)
    pool4 = pooling(conv12)
    
    conv13 = convdic(pool4)
    conv14 = convdic(conv13)
    conv5_3 = convdic(conv14)
    
    print(len(conv5_3))
    print(len(conv5_3[5][5]))
    print((80,80) in conv5_3[5][5])
    print((40,40) in conv5_3[5][5])
    
    return conv5_3
    #Remaining layers which do not go through processing:
    #conv16 = convdic(conv15)
    #pool5 = pooling(conv16)

def mapBack(reg):
    count = 0
    for key, tuplet in reg.items():
        for keyz, setlet in tuplet.items():
            for itemlet in setlet:
                count += 1
    print('{} items in reg'.format(count))
    
    alphaMap = {}
    for i in range(224):
        alphaMap[i] = {}
        for j in range(224):
            alphaMap[i][j] = set()
    
    for i in range(14):
        for j in range(14):
            for tupleSet in reg[i][j]:
                alphaMap[tupleSet[0]][tupleSet[1]].add((i,j))
    print(len(alphaMap)) 
    print(len(alphaMap[5][5]))
    print(alphaMap[5][5])
    
    count = 0
    for key, tuplet in alphaMap.items():
        for keyz, setlet in tuplet.items():
            for itemlet in setlet:
                count += 1
    print('{} items in alphamap'.format(count))
    '''
    
    
        '''
    with open('alphaMap.pkl', 'wb') as f:
        pickle.dump(alphaMap, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to {}'.format('alphaMap.pkl'))

    
if __name__ == '__main__':
    reg = reVGGNet19()
    mapBack(reg)