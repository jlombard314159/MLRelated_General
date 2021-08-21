from Python.Modeling.modelingGeneralPurpose import convertCovarGroup, wideCovarToLong
import glob
import pandas as pd
from Python.DataAssimilation.readDBF import *
from Python.Modeling.exploratoryModeling import * 
from Python.Modeling.randomForest import *
from Python.Modeling.modelingGeneralPurpose import *

#Note: multiple complaints from vs code
#Run code with python -m main.py (gives a warning)

def lassoModels():

    dbfPath = 'P:/102 - USGS/102-88 - Sagebrush WY/GIS/Input/Zonals/'

    listOfStuff = glob.glob(dbfPath + "/*.dbf")
 
    # zonalDF = iterateDBF(listOfDBF=listOfStuff)

    localPath = 'C:/Users/jlombardi/Documents/GitLabCode/'
    # # coeffData = readInCoeff(csvFile = localPath + 
    # #                         'sagecastr/output/CoreAreaClimateEffectsMeans.csv')

    # # finalData = mergeDataByID(coeffData = coeffData,
    # #                             dbfData=zonalDF)

    finalData = pd.read_pickle(localPath + 'sagecastr/exploratoryAnalysis/Python/output/workspace/modelingData.pk1')
    
    # # outputModelDFCSV(fullData=finalData, savePath=imageSavePath + 'modelingDF.csv')

    imageSavePath = 'C:/Users/jlombardi/Documents/GitLabCode/sagecastr/exploratoryAnalysis/Python/output/'

    modelingDF, responseData = createModelingData(fullData = finalData)

    alphas = alphaGeneratorLogSpace(seqStart=-12, seqEnd=2, numAlphas=300)

    standardizedCovariateDF = transformCovariates(covariateDF=modelingDF)

    for responseKey, responseDF in responseData.items():

        allResults = modelLassoCV(covariateDF=standardizedCovariateDF, responseData=responseDF, alphas= alphas)

        allCoeffDF = addNamesToCoefficients(coefficients = allResults['topCoeff'], coefficientLabels=list(modelingDF.columns))

        nonZeroCoeffDF = subsetCoeffToNonZero(allCoeffDF)

        varyAlphaCoeffList = fitLassoVaryAlpha(covariateData=standardizedCovariateDF,responseData=responseDF, alphasList = allResults['allAlphas'])

        createLambdaCVPlot(coeffList = varyAlphaCoeffList, alphas = allResults['allAlphas'],
            savePath=imageSavePath, modelAppender=responseKey, optimalLambda=allResults['topAlpha'])

        mseForPlot = getAverageMSE(allResults['cvInfoForPlotting'])

        createCVScorePlot(cvParameters=mseForPlot, alphas=allResults['allAlphas'], savePath = imageSavePath, 
            modelAppender=responseKey , optimalLambda=allResults['topAlpha'])

        # #Add one column
        nonZeroCoeffDF['rSquared'] = allResults['rSquared']

        outputNonZeroCoeff(nonZeroCoeff=nonZeroCoeffDF, savePath = imageSavePath ,
            modelAppender=responseKey)

        print('computation done for ' + responseKey)

def classifierModels():

    imageSavePath = 'C:/Users/jlombardi/Documents/GitLabCode/sagecastr/exploratoryAnalysis/Python/output/'
    localPath = 'C:/Users/jlombardi/Documents/GitLabCode/'

    finalData = pd.read_pickle(localPath + 'sagecastr/exploratoryAnalysis/Python/output/workspace/modelingData.pk1')

    estimatorGrid = createEstimatorSelection(models_classifier,params_classifier)

    modelingDF, responseData = createModelingData(fullData = finalData)

    modelingDF, responseData = shuffleData(modelingDF,responseData)

    responseData = convertToClassification(responseData)

    standardizedCoef = transformCovariates(modelingDF)

    estimatorResults = fitEstimatorSelection(estimatorSelect=estimatorGrid,
        covariates=standardizedCoef, response=[responseData['PrecipEffect'],responseData['TempEffect']], 
        scoring='f1',label = ['precip','temp'])

    #-----------------------
    #Below is manual, for now
    allScores = estimatorResults['precip']['allScores']
    allScores[allScores['estimator'] == 'LogReg']


    # finalModel = RandomForestClassifier(max_depth=3, n_estimators=4)
    finalModel = LogisticRegression(penalty = 'l2',C=0.0001,solver='liblinear')
    finalModel_temp = LogisticRegression(penalty = 'l1',C = 2,solver = 'liblinear')
    # finalModel_temp = BaggingClassifier(n_estimators=1000)

    finalModel.fit(modelingDF, responseData['PrecipEffect'])
    finalModel_temp.fit(modelingDF, responseData['TempEffect'])


    precipCoef = addNamesToCoefficients(finalModel.coef_.squeeze(),coefficientLabels=list(modelingDF.columns))
    precipNonZeroCoef = subsetCoeffToNonZero(precipCoef)

    tempCoef = addNamesToCoefficients(finalModel_temp.coef_.squeeze(),coefficientLabels=list(modelingDF.columns))
    tempNonZeroCoef = subsetCoeffToNonZero(tempCoef)

    tempPred = leaveOneOutModeling(standardizedCoef,responseData['TempEffect'],finalModel_temp)

    tempPreds = pd.DataFrame(data = {'actual':responseData['TempEffect'], 'prediction': tempPred})
    modelAccuracies(tempPreds)

    #TODO: maybe implement below as a fnction
    temp = tempNonZeroCoef.iloc[0]
    temp = abs(temp)
    temp.sort_values

#----------------------------------
    #Below is all commnted out as it is for RF specifically
    # precipImportances = featureImportance(model=finalModel,columnNames=list(modelingDF.columns))

    # tempImportances = baggingFeatureImportance(baggingModel=finalModel_temp,covarNames=list(modelingDF.columns))

    # topTempImportance = selectHighestImportances(tempImportances, topN=10)
    # topPrecipImportance = selectHighestImportances(precipImportances,topN=10)
    
    # importancePlot(topTempImportance,savePath=imageSavePath,modelAppender=', Temperature')
    # importancePlot(topPrecipImportance,savePath=imageSavePath,modelAppender=', Precip')

    # createTreePlot(oneTree = finalModel.estimators_[0], columnLabels=list(modelingDF.columns),
    #     classNames = ['negative precip','positive precip'], fileName = 'treePrecip')

    # createTreePlot(oneTree = finalModel_temp.estimators_[0], columnLabels=list(modelingDF.columns),
    #     classNames = ['negative temp','positive temp'], fileName = 'treeTemp')


    # #----------------------------------------------------------
    # #Sensitivity plots
    # sensData = createSensitivityData(covariateData=modelingDF, columnToVary=topTempImportance['covariate'].iloc[0])
    # predSens = modelSensitivityData(sensData, finalModel_temp)

    # calculateBreakPoint(predSens,topTempImportance['covariate'].iloc[0])

    # outputDict = {}
    # for index in range(0,topTempImportance.shape[0],1):

    #     sensData = createSensitivityData(covariateData=modelingDF, columnToVary=topTempImportance['covariate'].iloc[index])
    #     predSens = modelSensitivityData(sensData, finalModel_temp)

    #     breakPointDict = calculateBreakPoint(predSens,topTempImportance['covariate'].iloc[index])
    #     outputDict.update(breakPointDict)


    return None


def averageByDepthModels():

    imageSavePath = 'C:/Users/jlombardi/Documents/GitLabCode/sagecastr/exploratoryAnalysis/Python/output/'
    localPath = 'C:/Users/jlombardi/Documents/GitLabCode/'

    finalData = pd.read_pickle(localPath + 'sagecastr/exploratoryAnalysis/Python/output/workspace/modelingData.pk1')

    estimatorGrid_depth = createEstimatorSelection(models_classifier,params_classifier)

    modelingDF, responseData = createModelingData(fullData = finalData)

    avgDepth = wideCovarToLong(modelingDF)

    additionalCoeffData = readInCoeff(localPath + '/sagecastr/Output/environ_sensitivity_summary.csv')

    additionalCoeffData.drop(['NAME','simpName','sensPr','sensTmean','sensBoth','Unnamed: 0'], axis=1, inplace=True)
    additionalCoeffData['ID'] = additionalCoeffData['ID'] - 1
    
    avgDepth = convertCovarGroup(avgDepth)

    finalAvgDF = averageByDepth(avgDepth)

    depthModelingDF = depthLongToWide(finalAvgDF)

    depthModelingDF = mergeDataByID(depthModelingDF,additionalCoeffData)

    depthModelingDF, responseData = shuffleData(depthModelingDF,responseData)

    responseData = convertToClassification(responseData)
    depthModelingDF.drop(['ID'],axis=1,inplace=True)

    depthModelingDF = filterOutColumn(depthModelingDF)

    depthModelingDF = filterOutColumn(depthModelingDF,pattern="tr")

    standardizedCoef = transformCovariates(depthModelingDF)


    estimatorResults_depth = fitEstimatorSelection(estimatorSelect=estimatorGrid_depth,
        covariates=standardizedCoef, response=[responseData['PrecipEffect'],responseData['TempEffect']], 
        scoring='f1',label = ['precip','temp'])

    #-----------------------
    #Below is manual, for now
    allScores = estimatorResults_depth['temp']['allScores']
    allScores[allScores['estimator'] == 'LogReg']


    finalModel_temp = LogisticRegression(penalty = 'l1',C = 0.5,solver = 'liblinear')
    finalModel_temp.fit(standardizedCoef, responseData['TempEffect'])

    tempCoef = addNamesToCoefficients(finalModel_temp.coef_.squeeze(),coefficientLabels=list(depthModelingDF.columns))
    tempNonZeroCoef = subsetCoeffToNonZero(tempCoef)

    tempPred = leaveOneOutModeling(standardizedCoef,responseData['TempEffect'],finalModel_temp)

    tempPreds = pd.DataFrame(data = {'actual':responseData['TempEffect'], 'prediction': tempPred})
    temp = modelAccuracies(tempPreds)

    tempNonZeroCoef.to_csv(imageSavePath + 'topLogRegModel_temperature.csv')

    return None