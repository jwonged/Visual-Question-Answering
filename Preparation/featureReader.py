import numpy as np
import json

def loadTextFile():
	arrays = np.loadtxt('output.csv').tolist()
	for imageF in arrays:
		imageF.insert(0, 'name')
		print(imageF)
	print(arrays)
	for item in arrays:
		print(len(item))
	print(len(arrays))

def loadFeatureFromID(imgID):
	with open('outputData.json') as jsonFile:
		featureData = json.load(jsonFile)
		print(featureData[imgID][0])

def printJSONfile():
	with open('outputData.json') as jsonFile:
		print(json.dumps(json.load(jsonFile), indent=4, sort_keys=True))

