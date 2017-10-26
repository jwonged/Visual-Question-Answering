import numpy as np
import json

def loadTextFile():
	arrays = np.loadtxt('output.csv').tolist()
	for imageF in arrays:
		imageF.insert(0, 'name')
		print(imageF)
	print('--------------------------------------------------')
	print(arrays)
	for item in arrays:
		print(len(item))
	print(len(arrays))

def loadJSONfile():
	with open('outputData.json') as jsonFile:
		featureData = json.load(jsonFile)
		print(featureData['VQAset'][1]['name'])
		#print(json.dumps(featureData, indent=4, sort_keys=True))
