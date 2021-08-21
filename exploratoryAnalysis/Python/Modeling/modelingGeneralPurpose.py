import pandas as pd
import numpy as np

def leaveOneOutModeling(covariateData, response,model):


    if isinstance(covariateData,pd.DataFrame): 
        covariateData = covariateData.to_numpy()

    rowList = np.arange(0,covariateData.shape[0],1)

    predictions = []
    for index, rowData in enumerate(covariateData):
        indexToUse = np.setdiff1d(rowList,index)
        modelData = covariateData[list(indexToUse),:]
        respData = response.drop(index)
        model.fit(modelData,respData)

        rowDF = pd.DataFrame(rowData).transpose()
        predictions.append(model.predict(rowDF)[0])

    return predictions

def removeDepth(covarLabel):

    splitUnder = covarLabel.split('_')
    separator = "_"
    covarQuantileOnly = separator.join([splitUnder[0],splitUnder[1]])

    return covarQuantileOnly

def wideCovarToLong(covariateData,var_name = 'covariate',value_name ='value'):
    covariateData['ID'] = covariateData.index
    longDF = pd.melt(covariateData, id_vars='ID', var_name=var_name, value_name=value_name)

    return longDF

def convertCovarGroup(covariateData):

    covarAvgCol = map(removeDepth,covariateData['covariate'])
    covariateData['covarAvgdDepth'] = pd.Series(covarAvgCol)

    return covariateData

def averageByDepth(covariateData,groupCol = ['ID','covarAvgdDepth']):

    avgByDepth = covariateData.groupby(groupCol).mean()
    avgByDepth.reset_index(inplace=True)

    return avgByDepth

def depthLongToWide(avgDepthCovariate, index = 'ID'):

    depthModelingDF = avgDepthCovariate.pivot(index = index,columns = 'covarAvgdDepth')
    actualCovar = [x[1] for x in depthModelingDF.columns]
    depthModelingDF.columns = actualCovar  
    depthModelingDF.reset_index(level=0, inplace=True)  

    return depthModelingDF

def filterOutColumn(covariateData,pattern = 'mo'):

    toRemove = list(covariateData.filter(regex=pattern))
    covariateData = covariateData[covariateData.columns.drop(toRemove)]

    return covariateData