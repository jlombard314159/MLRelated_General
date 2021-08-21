from numpy.lib.arraysetops import setdiff1d
from pandas.core.frame import DataFrame
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def createModelingData(fullData, 
    nonCovariateColumns = ['ID','Name','NameNoSep','Abbreviation','PrecipEffect','TempEffect'],
    precipCovar = 'PrecipEffect', tempCovar = 'TempEffect'):

    modelingDF = fullData.drop(nonCovariateColumns, axis=1)

    responseDict = {precipCovar: fullData[precipCovar],
        tempCovar: fullData[tempCovar]}


    return modelingDF, responseDict

def alphaGeneratorLogSpace(seqStart, seqEnd, numAlphas):

    alphaSeq = np.logspace(seqStart, seqEnd, numAlphas)

    return alphaSeq

def transformCovariates(covariateDF: pd.DataFrame) -> pd.DataFrame:
    
    scaler = StandardScaler()
    covariateStandard = scaler.fit_transform(covariateDF)

    return covariateStandard 

def modelLassoCV(covariateDF, responseData, alphas):

    lassoAlpha = LassoCV(alphas = alphas)
    lassoAlpha.fit(X=covariateDF, y=responseData)

    cvInfo = lassoAlpha.mse_path_

    alphas = lassoAlpha.alphas_

    topAlpha = lassoAlpha.alpha_

    topRSquared = lassoAlpha.score(X=covariateDF, y=responseData)
    topCoeff = lassoAlpha.coef_

    results = {'rSquared':topRSquared,
    'topAlpha':topAlpha,
    'cvInfoForPlotting':cvInfo,
    'allAlphas':alphas,
    'topCoeff':topCoeff}

    return results

def fitLassoModel(modelingData, responseData, topAlpha):


    lassoFit = Lasso(alpha = topAlpha)
    lassoFit.fit(X= modelingData, y=responseData)

    rSquared = lassoFit.score(X=modelingData, y=responseData)

    coeffsLasso = lassoFit.coef_

    return coeffsLasso, rSquared

def addNamesToCoefficients(coefficients, coefficientLabels):

    modelDataFrame = pd.DataFrame(columns = coefficientLabels)

    modelDataFrame.loc[0] = coefficients

    return modelDataFrame

def subsetCoeffToNonZero(coefficientDF, removalThreshold = 0.0000001):

    filterNonZero = (np.abs(coefficientDF) > removalThreshold).any()

    nonZeroDF = coefficientDF.loc[:, filterNonZero]

    return nonZeroDF

def fitLassoVaryAlpha(covariateData, responseData, alphasList):

    coeffList = []
    for alpha in alphasList:
        coeffIter, _ = fitLassoModel(modelingData = covariateData, responseData=responseData, topAlpha = alpha)
        coeffList.append(coeffIter)

    return coeffList

def createLambdaCVPlot(coeffList, alphas, savePath, modelAppender, optimalLambda):

    ax = plt.gca()
    ax.plot(alphas, coeffList)
    ax.set_xscale('log')

    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.axvline(x=optimalLambda, color = 'black', linestyle = 'dashed', alpha = 0.5)
    plt.savefig(savePath + "/" + modelAppender + "_LassoAlpha.png")
    plt.clf()
    
    return None

def getAverageMSE(mseData):

    avgdData = mseData.mean(axis=1)

    return avgdData

def createCVScorePlot(cvParameters, alphas,savePath, modelAppender, optimalLambda):

    ax = plt.gca()
    ax.plot(alphas, cvParameters)
    ax.set_xscale('log')

    plt.xlabel('alpha')
    plt.ylabel('MSE')
    plt.title('Lasso MSE as a function of the regularization')
    plt.axis('tight')
    plt.axvline(x=optimalLambda, color = 'black', linestyle = 'dashed', alpha=0.5)
    plt.savefig(savePath + "/" + modelAppender + "_LassoCVScore.png")
    plt.clf()

    return None

def outputNonZeroCoeff(nonZeroCoeff,savePath,modelAppender):

    nonZeroCoeff.to_csv(savePath + modelAppender + '.csv')

    return None

def outputModelDFCSV(fullData, savePath):

    fullData.to_csv(savePath)

    return None
