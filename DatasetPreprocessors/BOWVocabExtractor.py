import json
import csv
from nltk import word_tokenize

class BOWVocabExtractor(object):
    def getAllQuestionDetails(self, fileName):
        print('Reading' + fileName)
        fileData = open(fileName)
        json_data = json.load(fileData)
        return json_data['questions']
    
    #Private returns list of tokenized questions
    def getTokenizedQuestions(self, questionDetails):
        print('Tokenizing...')
        tokenizedQuestions = []
        rawQns = []
        count = 0
        for question in questionDetails:
            lowercaseQn = question['question'].lower()
            tokenizedQuestions.append(word_tokenize(lowercaseQn))
            rawQns.append(lowercaseQn)
            count = count + 1
            if (count%1000 == 0):
                print ('Questions tokenized: ' + str(count))

        print('Completed : ' + str(count))
        return tokenizedQuestions, rawQns
    
    #Returns the Bag-of-Words Vocab
    def createBOWVocab(self, fileName):
        print('Reading from ' + fileName)
        vocabSet = set()
        vocabBOW = []
        questionDetails = self.getAllQuestionDetails(fileName)
        tokenizedQuestions, rawQns = self.getTokenizedQuestions(questionDetails)

        for tokQn in tokenizedQuestions:
            for word in tokQn:
                if word != '?' and word not in vocabSet:
                    vocabSet.add(word)
                    vocabBOW.append(word)
        return vocabBOW, rawQns
    
def writeToFile(fileName, vocab, outputRawQnsFile, rawQns):
    print('Writing to file: ' + fileName)
    print('Number of words:' + str(len(vocab)))
    with open(fileName, 'w') as outputVocabFile:
        writer = csv.writer(outputVocabFile)
        writer.writerow(vocab)

    with open(outputRawQnsFile, 'w') as qnOut:
        for qn in rawQns:
            qnOut.write(qn + '\n')

if __name__ == "__main__":
    #json file
    inputQuestions = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/MultipleChoice_mscoco_train2014_questions.json'
    outputVocabFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/MultipleChoiceVocab.csv'
    outputRawQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/MultipleChoiceRaw.txt'

    vocabGetter = BOWVocabExtractor()
    vocab, rawQns = vocabGetter.createBOWVocab(inputQuestions)
    writeToFile(outputVocabFile, vocab, outputRawQnsFile,rawQns)
    print('Completed first file')

    oeinputQuestions = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json'
    oeVocabFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/OpenEndedVocab.csv'
    oeRawQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/OpenEndedRaw.txt'

    vocabGetter2 = BOWVocabExtractor()
    oevocab, oerawQns = vocabGetter2.createBOWVocab(oeinputQuestions)
    writeToFile(oeVocabFile, oevocab, oeRawQnsFile, oerawQns)
    print('Completed second file')


