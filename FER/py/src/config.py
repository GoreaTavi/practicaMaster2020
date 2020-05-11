import os
from enum import Enum

folders = ['Training', 'PrivateTest', 'PublicTest']
label_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
image_formats = [("JPEG", ".jpg"), ("PNG", ".png")]

currentDirectory = os.path.normpath(os.getcwd() + os.sep + os.pardir)
outputPathStandard48 = currentDirectory + "\\fer2013-48"
outputPathStandard71 = currentDirectory + "\\fer2013-71"

outputPathLogsXception = currentDirectory + "\\Logs" + "\\Xception"

pathDATAHOG = currentDirectory + "\\Data" + "\\SVM" + "\\HOG"
pathDATAFAST = currentDirectory + "\\Data" + "\\SVM" + "\\FAST"
pathDATAXception = currentDirectory + "\\Data" + "\\Xception"

pathModelsHOG = currentDirectory + "\\Models" + "\\SVM" + "\\HOG"
pathModelsFAST = currentDirectory + "\\Models" + "\\SVM" + "\\FAST"
pathModelsXception = currentDirectory + "\\Models" + "\\Xception"


class Emotion(Enum):
    __order__ = 'Angry Disgust Fear Happy Sad Surprise Neutral'
    Angry = 0
    Disgust = 1
    Fear = 2
    Happy = 3
    Sad = 4
    Surprise = 5
    Neutral = 6
