from posixpath import basename
from dbfread import DBF
import pandas as pd
import os


def readDBFFile(dbfFileLocation):

    dbfData = DBF(dbfFileLocation)

    return dbfData

def convertDBFPandas(dbfData):

    pdDataFrame = pd.DataFrame(iter(dbfData))

    return pdDataFrame

def subsetDBFDataFrame(dbfDataFrame, keepCols = ['ID','MEAN']):

    subsetData = dbfDataFrame[keepCols]

    return subsetData

def appendName(dbfDataFrame, appendName, originalValue = 'MEAN'):

    modifyDF = dbfDataFrame.rename({originalValue: appendName}, axis=1)
    
    return modifyDF

def combineDBFDF(dbfDF,dbfDFToAdd,colToMergeOn = 'ID'):

    if dbfDFToAdd.empty:
        return dbfDF

    mergedDF = pd.merge(dbfDF, dbfDFToAdd,
        how = 'inner', on = colToMergeOn,
        suffixes = ('','_delme'))

    columnToEdit = list(mergedDF.columns)[-1]
    toRemove = columnToEdit.removesuffix('_delme')
    changeDFColName = appendName(mergedDF, originalValue=columnToEdit,
        appendName = toRemove)

    return changeDFColName

def extractFileName(fileName):

    baseName = str(os.path.basename(os.path.normpath(fileName)))

    removeFileType = baseName.removesuffix('.dbf')

    return removeFileType

def iterateDBF(listOfDBF):

    toFillDF = pd.DataFrame()

    for oneFile in listOfDBF:

        fileName = extractFileName(oneFile)

        extractDBF = readDBFFile(oneFile)
        pdDF = convertDBFPandas(extractDBF)
        subsetDF = subsetDBFDataFrame(pdDF)
        modifiedDF = appendName(subsetDF,fileName)
        toFillDF = combineDBFDF(modifiedDF,toFillDF)

    return toFillDF

def readInCoeff(csvFile: str) ->pd.DataFrame:

    coeffData = pd.read_csv(csvFile)

    return coeffData

def mergeDataByID(coeffData, dbfData, colToMerge = 'ID'):

    mergedDF = pd.merge(coeffData, dbfData,
        how = 'inner', on = colToMerge)

    return mergedDF
