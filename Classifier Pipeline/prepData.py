# Importing the libraries
import pandas as pd
import pdb

dataSetPath = 'C:/Users/jlombardi/Documents/GitLabCode/nnplay/Classifier Pipeline/Data/AllBatCallData.csv'

dataset = pd.read_csv(dataSetPath)

def PrepData(data,namesToDrop = ['ModelCount','Species'],
             speciesToInfo = 'All', dataKey = 'DatasetType',
             columnWithLabel = 'BinaryResponse'):

    #First subset to species
    subsettedData = data[data.Species == speciesToInfo]

    #Now get rid of additional columns
    subsettedData = subsettedData.drop(namesToDrop,1)

    #Now subset into train/valid
    trainData = subsettedData[subsettedData[dataKey].isin(['Training'])]
    validData = subsettedData[subsettedData[dataKey].isin(['Validation'])]

    #Now split column wise
    trainLabel = trainData[columnWithLabel]
    validLabel = validData[columnWithLabel]

    #Now delete some columns
    trainData = trainData.drop([dataKey,columnWithLabel],1)
    validData = validData.drop([dataKey,columnWithLabel],1)

    #Conver to dummies
    trainData = pd.get_dummies(trainData,drop_first=True)
    validData = pd.get_dummies(validData,drop_first=True)

    return trainData.values, validData.values,\
           trainLabel.values, validLabel.values

train_X, valid_X, train_Y, valid_Y = PrepData(dataset,
                                              speciesToInfo='All')

#---------------------
#Do a transformation
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

def ShuffleXData(train_X, train_Y):
    train_X, train_Y = shuffle(train_X, train_Y)

    return train_X, train_Y

def scaleXData(dataTrain, dataValidate):

    sc = StandardScaler()
    X_train = sc.fit_transform(dataTrain)
    X_test = sc.transform(dataValidate)

    return X_train, X_test

train_X, train_Y = ShuffleXData(train_X, train_Y)
train_X, valid_X = scaleXData(train_X, valid_X)

