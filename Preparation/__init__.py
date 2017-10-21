import json
from nltk import word_tokenize

def getAllQuestionDetails(fileName):
    fileData = open(fileName)
    json_data = json.load(fileData)
    return json_data['questions']

def createBOWVocab(questionSet):
    vocabBOW = {}
    for question in questionSet:
        for word in word_tokenize(question):
            if word != '?':
                vocabBOW[word] = 0
    return vocabBOW

