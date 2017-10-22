import json
import csv
from nltk import word_tokenize

class BOWVocabExtractor(object):
    def getAllQuestionDetails(self, fileName):
        fileData = open(fileName)
        json_data = json.load(fileData)
        return json_data['questions']
    
    #returns list of tokenized questions
    def getTokenizedQuestions(self, questionDetails):
        tokenizedQuestions = []
        for question in questionDetails:
            tokenizedQuestions.append(word_tokenize(question['question'].lower()))
        return tokenizedQuestions
    
    #Returns the Bag-of-Words
    def createBOWVocab(self, fileName):
        vocabSet = set()
        vocabBOW = []
        questionDetails = self.getAllQuestionDetails(fileName)
        tokenizedQuestions = self.getTokenizedQuestions(questionDetails)
        for tokQn in tokenizedQuestions:
            for word in tokQn:
                if word != '?' and word not in vocabSet:
                    vocabSet.add(word)
                    vocabBOW.append(word)
        return vocabBOW
    
    def writeToFile(self, file, text):
        with open(file, 'w', newline='') as vocabFile:
            writer = csv.writer(vocabFile)
            writer.writerow(text)
