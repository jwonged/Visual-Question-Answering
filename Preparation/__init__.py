import json

def getAllQuestionDetails(fileName):
    fileData = open(fileName)
    json_data = json.load(fileData)
    return json_data['questions']